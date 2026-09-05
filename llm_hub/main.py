"""FastAPI-App: OpenAI-kompatibler Proxy mit Auto-Load, Modell-Verwaltung,
Download-Endpoints mit Fortschritt, und gemounteter MCP-Server unter /mcp."""
from __future__ import annotations

import json
import logging
import time
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import asyncio
import random
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from . import active_streams, capability_detector, config_editor, conversation_tracker, cost_tracker, downloader, nvfp4_quantizer, perf_tuner, process_manager, rag, request_queue, telemetry
from .auth import ApiKeyMiddleware
from . import catalog
from .catalog import list_cached_models
from .config import get_config
from .chat_dashboard import router as chat_dashboard_router
from .config_dashboard import router as config_dashboard_router
from .conversations_dashboard import router as conversations_dashboard_router
from .cost_dashboard import router as cost_dashboard_router
from .dashboard import router as dashboard_router
from .mcp_tools import mcp
from .ollama_compat import router as ollama_router
from .rag_dashboard import router as rag_dashboard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm_hub")

IDLE_CHECK_INTERVAL = 30
ORPHAN_CHECK_INTERVAL = 300  # 5 Minuten - Prozess-Scan ist billig, muss nicht so oft wie der Idle-Check laufen
# Fallback für GET /v1/models -> maxInputTokens/maxOutputTokens, wenn kein
# max_model_len/max_tokens bekannt ist (nicht registrierte, nur gecachte
# Modelle) - reiner Platzhalter für Clients, die diese Felder erwarten.
_V1_MODELS_DEFAULT_MAX_TOKENS = 4096


async def _idle_check_once() -> None:
    """Ein Durchlauf des Idle-Watchdogs - ausgelagert aus _idle_watchdog(),
    damit er isoliert (ohne die Endlosschleife/den 30s-Sleep) testbar ist."""
    cfg = get_config()
    now = time.time()

    if not cfg.idle_timeout_seconds:
        return
    for eng in list(process_manager.engines.values()):
        if now - eng.last_used > cfg.idle_timeout_seconds:
            logger.info("Idle-Timeout (%ss) erreicht, entlade %s", cfg.idle_timeout_seconds, eng.model)
            await process_manager.stop_engine(eng.model, reason="idle_timeout")


async def _idle_watchdog() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL)
        await _idle_check_once()


async def _reconcile_dead_engines() -> None:
    """Ergänzt reap_orphan_engines() (verwaiste, UNGETRACKTE Prozesse) um den
    umgekehrten Fall: ein getrackter EngineState, dessen Prozess von selbst
    gestorben ist (z.B. OOM-Kill durch den Kernel), OHNE dass stop_engine()
    dafür aufgerufen wurde. Ohne diesen Check bliebe das Dashboard bei
    state="ready" hängen, obwohl die Engine längst tot ist."""
    for eng in list(process_manager.engines.values()):
        if eng.process is not None and eng.process.returncode is not None:
            logger.warning(
                "Engine für '%s' ist unbemerkt beendet (exit=%s) - räume EngineState auf",
                eng.model, eng.process.returncode,
            )
            eng.last_error = f"Engine unerwartet beendet (exit={eng.process.returncode})."
            await process_manager.stop_engine(eng.model, reason="crashed")


async def _orphan_watchdog() -> None:
    """Periodischer Scan auf verwaiste vLLM-Engine-Prozesse (siehe
    process_manager.reap_orphan_engines) - fängt Fälle ab, die der Scan beim
    Dienststart nicht sieht, z.B. wenn während der Laufzeit manuell ein
    zusätzlicher, ungetrackter "vllm serve"-Prozess dieser Installation
    auftaucht."""
    while True:
        await asyncio.sleep(ORPHAN_CHECK_INTERVAL)
        try:
            await _reconcile_dead_engines()
            killed = await process_manager.reap_orphan_engines()
            if killed:
                logger.warning("Periodischer Scan: %d verwaiste(n) Engine-Prozess(e) beendet: %s", len(killed), killed)
        except Exception:
            logger.exception("Periodischer Scan nach verwaisten Engine-Prozessen fehlgeschlagen")


async def _auto_reload_last_model() -> None:
    """Ollama-artiges Verhalten (config.auto_reload_last_model, Default an):
    lädt beim Dienststart automatisch das zuletzt genutzte Modell nach. Läuft
    als eigener Background-Task (siehe lifespan()) - blockiert also NICHT den
    Serverstart selbst (uvicorn nimmt sofort Requests an), auch wenn der
    Kaltstart hier 1-5 Minuten dauert. ensure_loaded() hält dabei wie bei jedem
    anderen Kaltstart auch den Pool-Lock (siehe process_manager.py) - ein
    echter Request für ein ANDERES, noch nicht geladenes Modell muss in dieser
    Zeit entsprechend warten, das ist bestehendes Verhalten des Hot Pools und
    kein neues Problem durch diese Funktion."""
    cfg = get_config()
    if not cfg.auto_reload_last_model:
        return
    model = process_manager.load_last_active_model()
    if not model:
        return
    ok, reason = process_manager.is_model_enabled(cfg, model)
    if not ok:
        logger.info("Automatisches Nachladen von '%s' übersprungen: %s", model, reason)
        return
    logger.info("Lade zuletzt aktives Modell '%s' automatisch im Hintergrund nach...", model)
    try:
        await process_manager.ensure_loaded(model)
        logger.info("Automatisches Nachladen von '%s' abgeschlossen.", model)
    except Exception as e:
        logger.warning("Automatisches Nachladen von '%s' fehlgeschlagen: %s", model, e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_editor.load_config_with_fallback()
    # Vor allem anderen: verwaiste Engine-Prozesse aus einem vorherigen Absturz
    # dieses Manager-Prozesses beenden (siehe process_manager.reap_orphan_engines) -
    # sonst startet z.B. der Auto-Reload unten in denselben Engine-Port hinein,
    # den ein Zombie-Prozess von vorhin noch belegt.
    try:
        killed = await process_manager.reap_orphan_engines()
        if killed:
            logger.warning("Beim Start: %d verwaiste(n) Engine-Prozess(e) aus vorherigem Lauf beendet: %s", len(killed), killed)
    except Exception:
        logger.exception("Aufräumen verwaister Engine-Prozesse beim Start fehlgeschlagen")
    # Downloads, die beim letzten Beenden (z.B. 'systemctl restart' für ein
    # Update) noch liefen, automatisch fortsetzen statt sie stillschweigend zu
    # verlieren - siehe downloader.resume_pending_downloads()-Docstring.
    try:
        resumed = await downloader.resume_pending_downloads()
        if resumed:
            logger.info("Beim Start: %d unterbrochene(n) Download(s) fortgesetzt: %s", len(resumed), resumed)
    except Exception:
        logger.exception("Fortsetzen unterbrochener Downloads beim Start fehlgeschlagen")
    watchdog = asyncio.create_task(_idle_watchdog())
    orphan_watchdog = asyncio.create_task(_orphan_watchdog())
    auto_reload = asyncio.create_task(_auto_reload_last_model())
    async with mcp.session_manager.run():
        yield
    watchdog.cancel()
    orphan_watchdog.cancel()
    auto_reload.cancel()
    await process_manager.stop_engine(reason="shutdown")


app = FastAPI(title="LLM Hub", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())
# Vendorte Frontend-Bibliotheken (siehe static/vendor/datatables/README.md) -
# lokal statt CDN, damit die Dashboards auch ohne Internetzugang funktionieren.
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
app.include_router(dashboard_router)
app.include_router(ollama_router)
app.include_router(rag_dashboard_router)
app.include_router(config_dashboard_router)
app.include_router(cost_dashboard_router)
app.include_router(chat_dashboard_router)
app.include_router(conversations_dashboard_router)
# Reine ASGI-Middleware statt @app.middleware("http") - siehe auth.py
# Docstring: BaseHTTPMiddleware bricht Streaming-Responses (stream: true).
app.add_middleware(ApiKeyMiddleware)


@app.get("/health")
async def health():
    return {"status": "ok", "engines": [e.status() for e in process_manager.engines.values()]}


@app.get("/models")
async def list_models_endpoint():
    cfg = get_config()
    cached = set(await list_cached_models(cfg.resolved_hf_home()))
    out = []
    for name, mcfg in cfg.models.items():
        out.append({
            "model": name,
            "cached": name in cached,
            "loaded": process_manager.is_ready(name),
            "enabled": mcfg.enabled,
            "notes": mcfg.notes,
        })
    known = {m["model"] for m in out}
    for name in sorted(cached - known):
        out.append({
            "model": name,
            "cached": True,
            "loaded": process_manager.is_ready(name),
            "enabled": True,
            "notes": "Lokal gecacht, aber nicht in config.json registriert.",
        })
    out.sort(key=lambda m: m["model"].lower())
    return {"models": out, "default_model": cfg.default_model, "active": process_manager.loaded_models()}


@app.post("/models/pull")
async def pull_model_endpoint(body: dict):
    model = body.get("model")
    if not model:
        raise HTTPException(400, "'model' fehlt im Request-Body, z.B. {\"model\": \"org/name\"}.")
    job_id = await downloader.start_job(model, body.get("revision"), body.get("hf_token"))
    return {"job_id": job_id}


@app.get("/models/pull")
async def list_pull_jobs():
    return {"jobs": downloader.list_jobs()}


@app.get("/models/pull/{job_id}")
async def pull_status_endpoint(job_id: str):
    job = downloader.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannte job_id.")
    return job


@app.post("/models/pull/{job_id}/cancel")
async def cancel_pull_endpoint(job_id: str):
    """Bricht einen laufenden/wartenden Download ab (Dashboard-Button) - siehe
    downloader.cancel_job(). Die bereits heruntergeladenen Teil-Dateien bleiben
    liegen (huggingface_hub kann fortsetzen); wer den Platz sofort zurückwill,
    nutzt danach DELETE /models/{model}/cache."""
    ok = await downloader.cancel_job(job_id)
    if not ok:
        raise HTTPException(404, f"Download-Job '{job_id}' existiert nicht (mehr) oder läuft nicht mehr.")
    return {"ok": True}


@app.post("/models/{model:path}/load")
async def load_model_endpoint(model: str, background: bool = False):
    """Lädt ein Modell manuell (Dashboard-"Laden"-Button, Gegenstück zu
    .../unload). Standardmäßig (background=false, bisheriges Verhalten)
    blockiert der Aufruf bis das Modell bereit ist oder cfg.startup_timeout_
    seconds abläuft - bei großen Modellen (mehrere Minuten Kaltstart) ist ein
    so lange offen gehaltener HTTP-Request/Browser-Fetch unpraktisch.
    background=true startet den Kaltstart stattdessen als überwachten
    Hintergrund-Task (exakt wie main._auto_reload_last_model) und gibt sofort
    zurück - Fortschritt/Fertig/Timeout zeigt wie gewohnt der nächste
    WebSocket-Heartbeat (Loaded Models/Modell-Verlauf)."""
    if background:
        asyncio.create_task(_background_load(model))
        return {"status": "loading_started"}
    try:
        return await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        raise HTTPException(500, str(e))


async def _background_load(model: str) -> None:
    try:
        await process_manager.ensure_loaded(model)
        logger.info("Manuelles Laden von '%s' abgeschlossen.", model)
    except Exception as e:
        logger.warning("Manuelles Laden von '%s' fehlgeschlagen: %s", model, e)
        # Ohne diesen Eintrag würde ein Fehlschlag hier (z.B. "kein Platz im Hot
        # Pool" - schlägt fehl, BEVOR überhaupt ein EngineState existiert, also
        # ohne den normalen Verlaufseintrag aus process_manager.stop_engine())
        # dem Nutzer komplett unsichtbar bleiben - der "Laden"-Button meldet nur
        # "loading_started" und danach passiert scheinbar nichts. Modell-Verlauf
        # zeigt den Grund stattdessen sichtbar an, wie bei jedem anderen Fehler.
        process_manager.model_history.appendleft({
            "model": model,
            "loaded_at": None,
            "unloaded_at": time.time(),
            "duration_seconds": None,
            "reason": "failed_to_start",
            "error": str(e),
        })


@app.get("/models/{model:path}/detect_capabilities")
async def detect_model_capabilities_endpoint(model: str):
    """Best-effort Erkennung von Vision/Tool-Calling/Reasoning/Task aus dem
    lokal gecachten chat_template.jinja + config.json des Modells - genutzt
    vom Config-Editor ("Fähigkeiten automatisch erkennen"-Button), siehe
    capability_detector.py. Rein lesend, kein Ladevorgang nötig."""
    cfg = get_config()
    return capability_detector.detect_capabilities(model, cfg.resolved_hf_home())


@app.post("/models/{model:path}/perf_tune")
async def perf_tune_endpoint(model: str):
    """Startet den automatisierten Performance-Test (Dashboard-Button "🚀
    Auto-tune performance") als Hintergrund-Job - siehe perf_tuner.py für den
    genauen Ablauf und die Sicherheitsvorkehrungen (nur ein Job gleichzeitig,
    Original-Config wird immer wiederhergestellt). Läuft mehrere Minuten,
    daher Job-Polling wie bei Downloads statt einer blockierenden Antwort."""
    other = nvfp4_quantizer.get_current_job()
    if other is not None:
        raise HTTPException(
            409,
            f"Es läuft gerade eine NVFP4-Quantisierung (für '{other['model']}') - beides braucht die GPU "
            f"exklusiv, bitte warten, bis sie fertig ist.",
        )
    try:
        job_id = await perf_tuner.start_job(model)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id}


@app.get("/models/perf_tune/{job_id}")
async def perf_tune_status_endpoint(job_id: str):
    job = perf_tuner.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannte job_id.")
    return job


@app.post("/models/perf_tune/{job_id}/cancel")
async def cancel_perf_tune_endpoint(job_id: str):
    ok = await perf_tuner.cancel_job(job_id)
    if not ok:
        raise HTTPException(404, f"Performance-Test-Job '{job_id}' existiert nicht (mehr) oder läuft nicht mehr.")
    return {"ok": True}


@app.post("/models/quantize/nvfp4")
async def quantize_nvfp4_endpoint(body: dict):
    """Startet die Selbst-Quantisierung eines HF-Modells zu NVFP4 als
    Hintergrund-Job (Dashboard-Button "NVFP4 quantisieren" im Config-Editor) -
    siehe nvfp4_quantizer.py für den genauen Ablauf. Läuft je nach Modellgröße
    viele Minuten (Docker-Image + Model-Optimizer-Install + Download +
    Kalibrierung), daher Job-Polling wie bei Downloads/Auto-Tune statt einer
    blockierenden Antwort."""
    model = body.get("model")
    if not model:
        raise HTTPException(400, "'model' fehlt im Request-Body, z.B. {\"model\": \"org/name\"}.")
    other = perf_tuner.get_current_job()
    if other is not None:
        raise HTTPException(
            409,
            f"Es läuft gerade ein Performance-Test (für '{other['model']}') - beides braucht die GPU "
            f"exklusiv, bitte warten, bis er fertig ist.",
        )
    try:
        job_id = await nvfp4_quantizer.start_job(
            model, body.get("tp", 1), body.get("hf_token"), bool(body.get("upgrade_transformers", False)),
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id}


@app.get("/models/quantize/nvfp4/{job_id}")
async def quantize_nvfp4_status_endpoint(job_id: str):
    job = nvfp4_quantizer.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannte job_id.")
    return job


@app.post("/models/quantize/nvfp4/{job_id}/cancel")
async def cancel_quantize_nvfp4_endpoint(job_id: str):
    ok = await nvfp4_quantizer.cancel_job(job_id)
    if not ok:
        raise HTTPException(404, f"Quantisierungs-Job '{job_id}' existiert nicht (mehr) oder läuft nicht mehr.")
    return {"ok": True}


@app.post("/models/{model:path}/unload")
async def unload_model_endpoint(model: str):
    if model in process_manager.engines:
        await process_manager.stop_engine(model)
        return {"status": "unloaded"}
    return {"status": "not_loaded", "currently_loaded": process_manager.loaded_models()}


@app.get("/models/{model:path}/cache_info")
async def model_cache_info_endpoint(model: str):
    """Größe der lokal heruntergeladenen Dateien eines Modells (falls
    vorhanden) - für den "Von Platte löschen"-Bestätigungsdialog im
    Dashboard, damit man vorher sieht, wie viel GB das eigentlich sind.
    Rein lesend, absichtlich getrennt vom Live-Snapshot (dashboard.py) - eine
    rekursive Verzeichnisgröße über evtl. hunderte GB soll nicht bei jedem
    WebSocket-Heartbeat für jedes Modell neu berechnet werden."""
    cfg = get_config()
    path = catalog.cache_dir_for(model, cfg.resolved_hf_home())
    if not path.exists():
        return {"cached": False, "size_bytes": 0, "path": str(path)}
    size = await asyncio.to_thread(catalog.dir_size_bytes, path)
    return {"cached": True, "size_bytes": size, "path": str(path)}


@app.delete("/models/{model:path}/cache")
async def delete_model_cache_endpoint(model: str):
    """Löscht die lokal heruntergeladenen Dateien eines Modells UNWIDERRUFLICH
    von der Platte (nicht nur die config.json-Registrierung - siehe
    Config-Editor "Remove", das nur den Eintrag entfernt). Greift für
    registrierte UND nur lokal gecachte, nicht (mehr) registrierte Modelle
    gleichermaßen. Verweigert, solange das Modell noch geladen ist oder gerade
    lädt - sonst würde der laufenden Engine buchstäblich der Boden unter den
    Füßen weggezogen."""
    if model in process_manager.engines:
        raise HTTPException(
            409,
            f"Modell '{model}' ist aktuell geladen (oder lädt gerade) - erst "
            f"entladen (Dashboard-Button oder POST .../unload), bevor die "
            f"Dateien von der Platte gelöscht werden können.",
        )
    cfg = get_config()
    path = catalog.cache_dir_for(model, cfg.resolved_hf_home())
    if not path.exists():
        raise HTTPException(404, f"Für Modell '{model}' sind lokal keine Dateien vorhanden.")
    freed = await asyncio.to_thread(catalog.delete_model_cache, model, cfg.resolved_hf_home())
    catalog.invalidate_cache(cfg.resolved_hf_home())
    catalog.invalidate_size_cache(model)
    return {"ok": True, "freed_bytes": freed}


# --- RAG (Retrieval-Augmented Generation) ---------------------------------
# Backend für die Dashboard-Seite /dashboard/rag. Dieselbe Logik ist auch über
# die rag_*-MCP-Tools nutzbar (siehe mcp_tools.py) - beide rufen rag.py auf.


@app.get("/rag/collections")
async def rag_list_collections_endpoint():
    try:
        return {"collections": await rag.list_collections()}
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.get("/rag/collections/{collection}/documents")
async def rag_list_documents_endpoint(collection: str):
    try:
        return {"collection": collection, "documents": await rag.list_documents(collection)}
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.get("/rag/collections/{collection}/documents/{document_id}/chunks")
async def rag_document_chunks_endpoint(collection: str, document_id: str):
    try:
        chunks = await rag.get_document_chunks(collection, document_id)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))
    return {"collection": collection, "document_id": document_id, "chunks": chunks}


@app.post("/rag/collections/{collection}/text")
async def rag_add_text_endpoint(collection: str, body: dict):
    text = body.get("text")
    if not text:
        raise HTTPException(400, "'text' fehlt im Request-Body.")
    try:
        return await rag.add_text(collection, text, source=body.get("source", "text"))
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.post("/rag/collections/{collection}/file")
async def rag_add_file_endpoint(collection: str, body: dict):
    path = body.get("path")
    if not path:
        raise HTTPException(400, "'path' fehlt im Request-Body.")
    try:
        return await rag.add_file(collection, path)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/rag/collections/{collection}/search")
async def rag_search_endpoint(collection: str, body: dict):
    query = body.get("query")
    if not query:
        raise HTTPException(400, "'query' fehlt im Request-Body.")
    try:
        results = await rag.search(collection, query, int(body.get("top_k", 5)))
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))
    return {"collection": collection, "results": results}


@app.delete("/rag/collections/{collection}/documents/{document_id}")
async def rag_delete_document_endpoint(collection: str, document_id: str):
    try:
        return await rag.delete_document(collection, document_id)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.delete("/rag/collections/{collection}")
async def rag_delete_collection_endpoint(collection: str):
    try:
        return await rag.delete_collection(collection)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


# --- Config-Editor (/dashboard/config) -------------------------------------
# Backend für die Config-Editor-Seite: liest/schreibt config.json über
# config_editor.py (Validierung + automatisches Backup + Live-Übernahme ohne
# Neustart für alles außer host/port, siehe dortige Docstrings).

# Felder, die nur beim Neustart des Manager-Prozesses selbst greifen (der
# Bind-Socket von uvicorn) - alles andere liest jeder Request/jeder neue
# Engine-Start ohnehin frisch über get_config(), ein Restart ist dafür nicht
# nötig.
RESTART_REQUIRED_FIELDS = {"host", "port"}


@app.get("/config")
async def get_config_endpoint():
    cfg = get_config()
    return {
        "config": cfg.model_dump(),
        "startup_warning": config_editor.startup_warning,
        # Vom Dashboard-Editor bei GET gemerkt und beim POST unten wieder
        # mitgeschickt (Header X-Config-Fingerprint) - erkennt, ob config.json
        # zwischen Laden und Speichern von anderer Stelle geändert wurde (z.B.
        # eine automatische Selbstkorrektur oder ein zweiter offener Tab), damit
        # ein Save aus einem inzwischen veralteten Editor-Zustand diese Änderung
        # nicht stillschweigend überschreibt. Siehe config_editor.save_config().
        "fingerprint": config_editor.config_fingerprint(cfg),
    }


@app.post("/config")
async def save_config_endpoint(body: dict, x_config_fingerprint: Optional[str] = Header(None)):
    old_dump = get_config().model_dump()
    try:
        new_cfg, backup_name = config_editor.save_config(
            body, expected_fingerprint=x_config_fingerprint
        )
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    except config_editor.ConfigConflictError as e:
        raise HTTPException(409, str(e))
    new_dump = new_cfg.model_dump()
    restart_recommended = any(old_dump.get(f) != new_dump.get(f) for f in RESTART_REQUIRED_FIELDS)
    return {"ok": True, "backup": backup_name, "restart_recommended": restart_recommended}


@app.get("/config/backups")
async def list_config_backups_endpoint():
    return {"backups": config_editor.list_backups()}


@app.post("/config/restore")
async def restore_config_backup_endpoint(body: dict):
    filename = body.get("filename")
    if not filename:
        raise HTTPException(400, "'filename' fehlt im Request-Body.")
    try:
        new_cfg, backup_name = config_editor.restore_backup(filename)
    except FileNotFoundError:
        raise HTTPException(404, f"Backup '{filename}' nicht gefunden.")
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    return {"ok": True, "backup": backup_name, "config": new_cfg.model_dump()}


@app.post("/config/restart")
async def restart_service_endpoint():
    ok, message = config_editor.restart_service()
    if not ok:
        raise HTTPException(500, message)
    return {"ok": True, "message": message}


# --- Kostentracking (/dashboard/costs) --------------------------------------
# Fiktive Kosten pro Anfrage ggü. einer Cloud-API, siehe cost_tracker.py.
# Lese-/Live-Endpoints liegen bei der Seite selbst (cost_dashboard.py), analog
# zu dashboard.py - hier nur die state-ändernden Aktionen (Löschen/Reset).


@app.delete("/costs/{record_id}")
async def delete_cost_record_endpoint(record_id: str):
    removed = cost_tracker.delete_records([record_id])
    if not removed:
        raise HTTPException(404, f"Kostendatensatz '{record_id}' nicht gefunden.")
    return {"ok": True, "removed": removed}


@app.post("/costs/delete")
async def delete_cost_records_endpoint(body: dict):
    ids = body.get("ids") or []
    if not ids:
        raise HTTPException(400, "'ids' fehlt oder ist leer im Request-Body.")
    removed = cost_tracker.delete_records(ids)
    return {"ok": True, "removed": removed}


@app.post("/costs/reset")
async def reset_cost_records_endpoint():
    removed = cost_tracker.reset_all()
    return {"ok": True, "removed": removed}


# --- Konversations-Log (/dashboard/conversations) - siehe conversation_tracker.py -------

@app.delete("/conversations/{record_id}")
async def delete_conversation_record_endpoint(record_id: str):
    removed = conversation_tracker.delete_records([record_id])
    if not removed:
        raise HTTPException(404, f"Konversations-Datensatz '{record_id}' nicht gefunden.")
    return {"ok": True, "removed": removed}


@app.post("/conversations/delete")
async def delete_conversation_records_endpoint(body: dict):
    ids = body.get("ids") or []
    if not ids:
        raise HTTPException(400, "'ids' fehlt oder ist leer im Request-Body.")
    removed = conversation_tracker.delete_records(ids)
    return {"ok": True, "removed": removed}


@app.post("/conversations/reset")
async def reset_conversation_records_endpoint():
    removed = conversation_tracker.reset_all()
    return {"ok": True, "removed": removed}


# --- OpenAI-kompatibler Proxy mit Auto-Load -------------------------------

HOP_BY_HOP = {"host", "authorization", "content-length", "connection"}


_MAX_TOKENS_PATHS = {"chat/completions", "completions"}


def _apply_default_max_tokens(cfg, model: str, path: str, parsed_body, body: bytes) -> bytes:
    """Injiziert models.<model>.max_tokens (siehe config.py) in den
    Request-Body, falls konfiguriert UND der Client selbst weder max_tokens
    noch max_completion_tokens mitschickt - ein expliziter Client-Wunsch wird
    nie überschrieben. Reines Sicherheitsnetz gegen durchgehende
    Generierungen (siehe ModelConfig.max_tokens-Docstring), nicht gedacht als
    genereller Parameter-Zwang."""
    if parsed_body is None or path not in _MAX_TOKENS_PATHS:
        return body
    mcfg = cfg.models.get(model)
    if mcfg is None or mcfg.max_tokens is None:
        return body
    if "max_tokens" in parsed_body or "max_completion_tokens" in parsed_body:
        return body
    parsed_body["max_tokens"] = mcfg.max_tokens
    return json.dumps(parsed_body).encode("utf-8")


def _apply_default_repetition_penalty(cfg, model: str, path: str, parsed_body, body: bytes) -> bytes:
    """Injiziert models.<model>.repetition_penalty (siehe config.py) in den
    Request-Body, falls konfiguriert UND der Client selbst kein
    repetition_penalty mitschickt - ein expliziter Client-Wunsch wird nie
    überschrieben. Vorbeugung gegen Wiederholungsschleifen (kleinere
    Reasoning-Modelle wie Nemotron-3-Nano neigen dazu, siehe ModelConfig.
    repetition_penalty-Docstring)."""
    if parsed_body is None or path not in _MAX_TOKENS_PATHS:
        return body
    mcfg = cfg.models.get(model)
    if mcfg is None or mcfg.repetition_penalty is None:
        return body
    if "repetition_penalty" in parsed_body:
        return body
    parsed_body["repetition_penalty"] = mcfg.repetition_penalty
    return json.dumps(parsed_body).encode("utf-8")


# Sicherheitsnetz gegen Wiederholungsschleifen (siehe ModelConfig.
# repetition_detection-Docstring): prüft, ob das Ende des bisher gestreamten
# Texts aus demselben Abschnitt mehrfach hintereinander EXAKT besteht - das
# typische Muster, wenn ein (meist kleineres) Reasoning-Modell im
# Denkprozess hängen bleibt und denselben Satz/Satzteil endlos wiederholt.
#
# Wichtig: die Länge des sich wiederholenden Abschnitts (die "Periode") ist
# vorher nicht bekannt - ein Modell könnte einen 3-Zeichen-Tick genauso wie
# einen ganzen 200-Zeichen-Satz wiederholen. Ein naiver Ansatz mit fest
# gewählten Fenstergrößen verpasst praktisch jede Periode, die kein Vielfaches
# der gewählten Fenstergröße ist - deshalb wird hier JEDE Periodenlänge in
# einem plausiblen Bereich einzeln geprüft (billig genug: pro Aufruf nur
# einige hundert schnelle C-seitige String-Vergleiche über höchstens
# _REPETITION_LOOKBACK Zeichen). Kürzere Perioden verlangen mehr exakte
# Wiederholungen, bevor sie als Schleife gelten - ein zufälliges
# Zusammentreffen bei "und und" (Periode ~4) ist normal, bei einem ganzen
# wiederholten Satz (Periode >80) dagegen praktisch ausgeschlossen.
_REPETITION_LOOKBACK = 2000
_REPETITION_MIN_PERIOD = 3
_REPETITION_MAX_PERIOD = 400

# Automatischer Neuversuch nach erkannter Wiederholungsschleife (siehe
# gen()/_build_retry_body unten, Chat vom 2026-08-27): statt sofort mit einer
# Abbruch-Notiz zu beenden, startet der Manager bis zu _LOOP_RETRY_MAX
# frische Generierungsversuche im SELBEN SSE-Stream nach - der Client (VS
# Code o.ä.) sieht eine durchgehende Antwort, keinen Fehler/Timeout. Bewusst
# NICHT gepuffert (kein Zurückrollen des bereits gestreamten Schleifentexts -
# das würde die Zeit bis zum ersten sichtbaren Token bei JEDER Anfrage
# erhöhen, nicht nur bei tatsächlichen Schleifen) - der Schleifentext bleibt
# sichtbar, eine Notiz macht den Neuversuch kenntlich.
_LOOP_RETRY_MAX = 2

# Registry laufender Streams (rid -> httpx.Response, stream=True), damit
# POST /requests/{rid}/cancel (Dashboard-Button "Abbrechen" bei Active
# Requests) eine hängende Anfrage manuell beenden kann - z.B. wenn der
# eigentliche Client (VSCode o.ä.) längst nicht mehr zuhört, die Engine aber
# unbemerkt weiter generiert (siehe Chat vom 2026-08-26: genau dieser Fall
# trat beim "completely—"-Loop auf, wo die automatische Erkennung nicht
# griff), sowie für die automatische Disconnect-Erkennung (siehe
# active_streams.watch_disconnect(), Chat vom 2026-09-01). Ausgelagert in ein
# eigenes Modul, damit ollama_compat.py (api_chat) dieselbe Registry nutzen
# kann wie proxy_v1()/gen() hier - siehe active_streams.py-Docstring.


def _min_repeats_for_period(period: int) -> int:
    if period < 10:
        return 8
    if period < 30:
        return 6
    if period < 80:
        return 4
    return 3


def _has_repetition_loop(text: str) -> bool:
    tail = text[-_REPETITION_LOOKBACK:]
    n = len(tail)
    for period in range(_REPETITION_MIN_PERIOD, _REPETITION_MAX_PERIOD + 1):
        min_repeats = _min_repeats_for_period(period)
        needed = period * min_repeats
        if n < needed:
            # NICHT einfach abbrechen: min_repeats_for_period() ist eine
            # Treppenfunktion, "needed" also nicht streng monoton in period
            # (z.B. period=9→8 Wiederholungen=72 Zeichen, period=10→6
            # Wiederholungen=60 Zeichen - kleiner trotz größerer Periode).
            continue
        segment = tail[-needed:]
        pattern = segment[:period]
        if all(segment[i * period:(i + 1) * period] == pattern for i in range(min_repeats)):
            return True
    return False


def _build_loop_abort_chunk(model: str, field: str) -> bytes:
    """Baut einen synthetischen SSE-Abschluss-Chunk, der dem Client sichtbar
    mitteilt, WARUM der Stream vorzeitig endet - statt ihn einfach kommentarlos
    abzuschneiden (der Client würde sonst nur eine unerklärt abgebrochene
    Antwort sehen). `field` ist "content" oder das vom aktuellen Stream
    tatsächlich genutzte Denkprozess-Feld ("reasoning" oder, bei älteren
    vLLM-Versionen, "reasoning_content" - siehe Aufrufer), je nachdem wo die
    Schleife erkannt wurde - die Notiz landet im selben Feld, damit sie dort
    auftaucht, wo der Client gerade hinschaut (z.B. Reasoning-Box vs.
    normaler Antworttext)."""
    note = "\n\n[Automatisch abgebrochen: Wiederholungsschleife erkannt]"
    obj = {
        "id": "loop-abort",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {field: note}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(obj)}\n\ndata: [DONE]\n\n".encode("utf-8")


def _build_loop_retry_chunk(model: str, field: str, attempt: int) -> bytes:
    """Wie _build_loop_abort_chunk, aber OHNE finish_reason/[DONE] - der
    Stream läuft im selben SSE-Response mit einem frischen Generierungs-
    versuch weiter (siehe _LOOP_RETRY_MAX). Der bereits gestreamte
    Schleifentext lässt sich mitten im Stream nicht mehr zurückrollen, diese
    Notiz macht wenigstens sichtbar, dass gerade automatisch neu versucht
    wird, statt dass unvermittelt anderer Text weitergeht."""
    note = f"\n\n[Wiederholungsschleife erkannt, automatischer Neuversuch {attempt}/{_LOOP_RETRY_MAX} …]\n\n"
    obj = {
        "id": "loop-retry",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {field: note}, "finish_reason": None}],
    }
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _build_retry_body(parsed_body: Optional[dict], attempt: int) -> Optional[bytes]:
    """Baut den Request-Body für einen automatischen Neuversuch nach
    erkannter Wiederholungsschleife: dieselbe Anfrage, aber mit angehobenem
    repetition_penalty und frischem, zufälligem seed - sonst würde ein
    deterministisch samplendes Modell (temperature=0 oder vom Client fest
    gesetzter seed) exakt dieselbe Schleife erneut produzieren. Gibt None
    zurück, wenn der ursprüngliche Body kein JSON war (dann ist kein
    sinnvoller Neuversuch möglich, siehe Aufrufer)."""
    if parsed_body is None:
        return None
    retry_body = dict(parsed_body)
    base_penalty = retry_body.get("repetition_penalty") or 1.0
    retry_body["repetition_penalty"] = min(1.5, base_penalty + 0.15 * attempt)
    retry_body["seed"] = random.randint(0, 2**31 - 1)
    return json.dumps(retry_body).encode("utf-8")


def _finish_and_record(
    rid: str,
    status: str,
    prompt_tokens=None,
    completion_tokens=None,
    *,
    request_messages: Optional[list] = None,
    request_prompt: Optional[str] = None,
    request_params: Optional[dict] = None,
    output_content: Optional[str] = None,
    output_reasoning: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> None:
    """telemetry.finish_request() + Persistieren fürs fiktive Kostentracking
    (cost_tracker.py) in einem Rutsch - auch für früh abgebrochene Anfragen
    (unbekanntes Modell, Ladefehler), damit die Kostenseite wirklich jede
    Anfrage zeigt (cost_usd bleibt dann None, siehe cost_tracker.compute_cost).

    request_messages/request_prompt werden NUR vom einzigen Aufrufer übergeben,
    der eine Anfrage tatsächlich bis zur Engine durchgereicht hat (gen() unten) -
    daran hängt zusätzlich das Persistieren des echten Konversationsinhalts
    (conversation_tracker.py, /dashboard/conversations). Alle anderen Aufrufer
    (unbekanntes Modell, Engine-Start fehlgeschlagen, noch in der Warteschlange
    abgebrochen) lassen diese Parameter bewusst weg - da wurde nie etwas
    generiert, es gibt keinen "Konversationsinhalt" zum Zeigen."""
    finished = telemetry.finish_request(rid, status, prompt_tokens, completion_tokens)
    if finished is not None:
        cost_tracker.record_request(
            model=finished["model"],
            path=finished["path"],
            started_at=finished["started_at"],
            finished_at=finished["finished_at"],
            status=finished["status"],
            prompt_tokens=finished.get("prompt_tokens"),
            completion_tokens=finished.get("completion_tokens"),
            user_agent=finished.get("user_agent"),
        )
        if request_messages is not None or request_prompt is not None:
            conversation_tracker.record_conversation(
                rid=rid,
                model=finished["model"],
                path=finished["path"],
                started_at=finished["started_at"],
                finished_at=finished["finished_at"],
                status=finished["status"],
                prompt_tokens=finished.get("prompt_tokens"),
                completion_tokens=finished.get("completion_tokens"),
                user_agent=finished.get("user_agent"),
                request_messages=request_messages,
                request_prompt=request_prompt,
                request_params=request_params,
                output_content=output_content or "",
                output_reasoning=output_reasoning or "",
                finish_reason=finish_reason,
            )


@app.post("/requests/{rid}/cancel")
async def cancel_request_endpoint(rid: str):
    """Bricht eine laufende Anfrage manuell ab (Dashboard-Button "Abbrechen"
    bei Active Requests) - deckt sowohl über proxy_v1() weitergeleitete
    Anfragen als auch das Ollama-Kompatibilitäts-Layer (ollama_compat.py
    /api/chat) ab, siehe active_streams.py. Vor dem 2026-09-01-Fix landete
    /api/chat nie in dieser Registry (kein Streaming-Aufruf dort) und der
    Button lief hier immer in die 404 "bereits beendet"."""
    if await active_streams.cancel(rid):
        return {"ok": True}
    # Noch nicht an ein Modell übergeben, sondern in der globalen
    # Concurrency-Warteschlange (siehe request_queue.py) - ein anderer
    # Abbruch-Pfad als bei einer bereits laufenden Anfrage, da hier noch
    # keine Upstream-Verbindung existiert, die man schließen könnte.
    if request_queue.cancel_waiting(rid):
        _finish_and_record(rid, "cancelled")
        return {"ok": True}
    raise HTTPException(404, f"Keine aktive, abbrechbare Anfrage mit id '{rid}' gefunden (evtl. bereits beendet).")


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request):
    cfg = get_config()
    body = await request.body()

    if path == "models" and request.method == "GET":
        cached = await list_cached_models(cfg.resolved_hf_home())
        names = sorted(set(cfg.models.keys()) | set(cached))
        # base_url zeigt auf DIESEN Manager (nicht auf die interne Engine, siehe
        # engine_host/engine_port oben) - aus dem eingehenden Request abgeleitet
        # statt aus cfg.host (der ist standardmäßig "0.0.0.0", also für einen
        # Client nutzlos), damit "url" unabhängig davon stimmt, über welche
        # Adresse (localhost/LAN-IP/Hostname) der Manager gerade erreicht wird.
        base_url = str(request.base_url).rstrip("/")
        data = []
        for n in names:
            mcfg = cfg.models.get(n)
            data.append({
                "id": n,
                "object": "model",
                "owned_by": "llm-hub",
                "name": n,
                "url": f"{base_url}/v1",
                # Inferenz-Backend, das den Prozess für dieses Modell startet
                # (siehe ModelConfig.engine / process_manager._build_command)
                # - "vllm" oder "sglang". Nur lokal gecachte, nicht in
                # config.json registrierte Modelle (mcfg is None) laufen
                # ausschließlich über vLLM (Auto-Detect kennt kein SGLang),
                # daher der Default.
                "engine": mcfg.engine if mcfg else "vllm",
                # toolCalling/vision/maxInputTokens spiegeln die Auto-Detect-
                # Werte aus capability_detector.py (siehe ModelConfig-Docstring
                # in config.py) - bei nur lokal gecachten, nicht registrierten
                # Modellen (mcfg is None) gibt es die nicht, dann konservative
                # Defaults.
                "toolCalling": bool(mcfg.enable_auto_tool_choice) if mcfg else False,
                "vision": bool(mcfg.vision) if mcfg else False,
                "maxInputTokens": mcfg.max_model_len if mcfg and mcfg.max_model_len else _V1_MODELS_DEFAULT_MAX_TOKENS,
                # maxOutputTokens = ModelConfig.max_tokens (Sicherheitsnetz-
                # Obergrenze, siehe _apply_default_max_tokens) - None (kein
                # Limit gesetzt) fällt ebenfalls auf den Default zurück.
                "maxOutputTokens": mcfg.max_tokens if mcfg and mcfg.max_tokens else _V1_MODELS_DEFAULT_MAX_TOKENS,
            })
        return {"object": "list", "data": data}

    model = None
    is_stream = False
    parsed_body: Optional[dict] = None
    if body:
        try:
            parsed_body = json.loads(body)
            model = parsed_body.get("model")
            is_stream = bool(parsed_body.get("stream"))
        except Exception:
            pass
    model = model or cfg.default_model
    if not model:
        raise HTTPException(400, "Kein 'model' im Request-Body und kein default_model in config.json konfiguriert.")

    rid = telemetry.start_request(model, path, is_stream, request.headers.get("user-agent"))

    # Cache-Scan nur, wenn wirklich nötig - im Normalfall (registriertes
    # Modell in config.json) spart das den Disk-I/O-Aufruf auf jeder einzelnen
    # Chat-Anfrage komplett, siehe catalog.py-Docstring.
    if model not in cfg.models and model not in await list_cached_models(cfg.resolved_hf_home()):
        _finish_and_record(rid, "error")
        raise HTTPException(
            404,
            f"Modell '{model}' ist unbekannt. Erst per POST /models/pull herunterladen "
            f"oder in config.json unter \"models\" registrieren.",
        )

    # Globale Concurrency-Grenze (siehe config.max_concurrent_requests/
    # queue_debounce_seconds, request_queue.py) - bewusst VOR ensure_loaded():
    # ein wartender Request soll noch keinen Modell-Kaltstart/-Wechsel
    # auslösen oder GPU-Zeit belegen, bis er tatsächlich dran ist. Ab hier bis
    # zum release() im finally von gen() (bzw. im except-Zweig unten bei
    # Fehlschlag vor dem Streaming) zählt diese Anfrage als "aktiv".
    await request_queue.acquire(rid)
    try:
        engine_status = await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        request_queue.release()
        _finish_and_record(rid, "error")
        raise HTTPException(503, str(e))
    telemetry.mark_ready(rid)

    body = _apply_default_max_tokens(cfg, model, path, parsed_body, body)
    body = _apply_default_repetition_penalty(cfg, model, path, parsed_body, body)
    if parsed_body is not None and path == "chat/completions":
        rag_result = await rag.apply_auto_rag(model, parsed_body.get("messages") or [])
        if rag_result:
            telemetry.mark_rag_used(rid, ", ".join(rag_result["collections"]), rag_result["hits"])
            body = json.dumps(parsed_body).encode("utf-8")

    target = f"http://{cfg.engine_host}:{engine_status['port']}/v1/{path}"
    client = httpx.AsyncClient(timeout=None)
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    req = client.build_request(request.method, target, content=body, headers=fwd_headers)
    try:
        upstream = await client.send(req, stream=True)
    except Exception:
        # Engine zwischen mark_ready() und hier weggebrochen (selten, aber
        # vorgekommen) - Slot muss trotzdem wieder frei werden, sonst bliebe
        # er dauerhaft belegt (siehe request_queue.acquire() oben).
        request_queue.release()
        _finish_and_record(rid, "error")
        raise
    result_status = "ok" if upstream.status_code < 400 else "error"
    active_streams.register(rid, upstream)

    # Für /dashboard/conversations (conversation_tracker.py): "messages" bei
    # chat/completions enthält bereits alle evtl. Änderungen von oben (Auto-
    # RAG-Kontext, Default-max_tokens/-repetition_penalty landen dagegen in
    # request_params, nicht in messages/prompt) - genau das, was tatsächlich
    # an die Engine ging, nicht nur das, was der Client ursprünglich schickte.
    # "prompt" deckt das ältere /v1/completions ab (kein messages-Array).
    request_messages = parsed_body.get("messages") if parsed_body is not None and path == "chat/completions" else None
    request_prompt = parsed_body.get("prompt") if parsed_body is not None and path == "completions" else None
    request_params = (
        {k: v for k, v in parsed_body.items() if k not in ("messages", "prompt", "model")}
        if parsed_body is not None and (request_messages is not None or request_prompt is not None)
        else None
    )

    async def gen():
        first_chunk = True
        full_body = bytearray()
        prompt_tokens = None
        completion_tokens = None
        status = result_status
        # Vollständiger (nicht wie content_buf/reasoning_buf unten pro
        # Retry-Versuch zurückgesetzter bzw. auf _REPETITION_LOOKBACK
        # gekürzter) Mitschnitt des tatsächlich an den Client gestreamten
        # Textes, fürs Konversations-Log (siehe _finish_and_record-Aufruf
        # unten) - unabhängig von der Wiederholungsschleifen-Erkennung.
        output_content_parts: list[str] = []
        output_reasoning_parts: list[str] = []
        finish_reason: Optional[str] = None
        # Wiederholungsschleifen-Erkennung (siehe _has_repetition_loop oben) -
        # nur bei gestreamten Antworten möglich, da wir sonst erst nach dem
        # kompletten (evtl. endlosen) Response irgendetwas sehen. mcfg ist
        # None bei nur lokal gecachten, nicht registrierten Modellen - dann
        # greift der Default (an), analog zu ModelConfig.repetition_detection.
        mcfg = cfg.models.get(model)
        detect_loop = is_stream and (mcfg.repetition_detection if mcfg is not None else True)
        current_upstream = upstream
        attempt = 0
        # Automatische Erkennung "Client ist weg" (siehe active_streams.
        # watch_disconnect() Docstring, Chat vom 2026-09-01) - läuft parallel
        # zum eigentlichen Lesen der Engine-Antwort und bricht active_streams
        # selbst ab, sobald uvicorn einen echten TCP-Disconnect meldet. Wird
        # im finally unten wieder beendet.
        disconnect_watcher = asyncio.create_task(active_streams.watch_disconnect(request, rid))
        try:
            # Äußere Schleife = automatische Neuversuche nach einer erkannten
            # Wiederholungsschleife (siehe _LOOP_RETRY_MAX oben) - jede
            # Iteration liest EINEN Generierungsversuch komplett durch, bei
            # aborted_loop mit noch verbleibenden Versuchen wird current_upstream
            # durch einen frischen Request ersetzt und die Schleife läuft
            # weiter, statt den Stream zu beenden.
            while True:
                sse_buffer = b""
                reasoning_buf = ""
                content_buf = ""
                aborted_loop = False
                abort_field = "content"
                async for chunk in current_upstream.aiter_raw():
                    if first_chunk:
                        telemetry.mark_first_token(rid)
                        first_chunk = False
                    if is_stream:
                        sse_buffer += chunk
                        while b"\n\n" in sse_buffer:
                            event, sse_buffer = sse_buffer.split(b"\n\n", 1)
                            for line in event.split(b"\n"):
                                if not line.startswith(b"data: "):
                                    continue
                                payload = line[len(b"data: "):].strip()
                                if payload == b"[DONE]":
                                    continue
                                try:
                                    obj = json.loads(payload)
                                except Exception:
                                    continue
                                choices = obj.get("choices") or []
                                delta = (choices[0].get("delta") or {}) if choices else {}
                                if choices and choices[0].get("finish_reason"):
                                    finish_reason = choices[0]["finish_reason"]
                                # Legacy /v1/completions-Chunks haben KEIN "delta"-
                                # Wrapper, sondern direkt choices[0].text - delta
                                # bleibt für die dort oben leer, die Wiederholungs-
                                # schleifen-Erkennung greift für diesen alten
                                # Endpunkt schon bisher nicht (siehe fehlendes
                                # "delta" hier), fürs Konversations-Log wird der
                                # Text trotzdem mitgeschnitten.
                                if choices and choices[0].get("text"):
                                    output_content_parts.append(choices[0]["text"])
                                if delta.get("content"):
                                    telemetry.increment_tokens(rid)
                                    output_content_parts.append(delta["content"])
                                    if detect_loop:
                                        content_buf = (content_buf + delta["content"])[-_REPETITION_LOOKBACK:]
                                        if _has_repetition_loop(content_buf):
                                            aborted_loop, abort_field = True, "content"
                                # reasoning/reasoning_content: separates Feld für den
                                # Denkprozess bei Modellen mit reasoning_parser (siehe
                                # config.json) - zählt NICHT in "content" hinein, deshalb
                                # eigener Zähler. Feldname je vLLM-Version unterschiedlich:
                                # aktuelle Versionen (siehe installierte 0.26.0,
                                # entrypoints/openai/chat_completion/protocol.py) liefern
                                # "reasoning", "reasoning_content" ist dort als deprecated
                                # markiert - live per curl gegen den echten Stream entdeckt
                                # (2026-08-28, siehe Chat): mit nur "reasoning_content"
                                # geprüft lief diese ganze Erkennung seit dem vLLM-Update
                                # ins Leere. abort_field übernimmt bewusst den Feldnamen,
                                # den DIESER Stream tatsächlich nutzt (nicht hart
                                # "reasoning_content"), damit die Abbruch-Notiz beim Client
                                # im selben Feld landet wie der bisherige Denkprozess-Text.
                                reasoning_field = "reasoning" if "reasoning" in delta else (
                                    "reasoning_content" if "reasoning_content" in delta else None
                                )
                                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                                if reasoning_delta:
                                    telemetry.increment_reasoning_tokens(rid)
                                    output_reasoning_parts.append(reasoning_delta)
                                    if detect_loop:
                                        reasoning_buf = (reasoning_buf + reasoning_delta)[-_REPETITION_LOOKBACK:]
                                        if _has_repetition_loop(reasoning_buf):
                                            aborted_loop, abort_field = True, reasoning_field or "reasoning"
                                if delta.get("tool_calls"):
                                    telemetry.mark_tool_call(rid)
                                usage = obj.get("usage")
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                                    # Live sichtbar machen, falls der Client stream_options.include_usage
                                    # gesetzt hat und usage schon vor dem letzten Chunk mitkommt.
                                    telemetry.update_partial_usage(rid, prompt_tokens, completion_tokens)
                    else:
                        full_body.extend(chunk)
                    yield chunk
                    if aborted_loop:
                        # Laufenden Read beenden (Engine erkennt das gleich als
                        # Client-Disconnect und bricht die Generierung dieses
                        # Versuchs ab) - was danach passiert, entscheidet der
                        # Retry-Block unten (Neuversuch oder endgültiger Abbruch).
                        break
                if not aborted_loop:
                    # Versuch normal zu Ende (kein Loop erkannt) - fertig, kein Retry.
                    break
                # aborted_loop kann nur True werden, wenn detect_loop bereits True
                # war (siehe oben) - kein zusätzlicher detect_loop-Check nötig.
                retry_body = _build_retry_body(parsed_body, attempt + 1)
                if attempt < _LOOP_RETRY_MAX and retry_body is not None:
                    attempt += 1
                    yield _build_loop_retry_chunk(model, abort_field, attempt)
                    logger.warning(
                        "Wiederholungsschleife erkannt, automatischer Neuversuch %d/%d: Modell '%s', Feld '%s' (rid=%s)",
                        attempt, _LOOP_RETRY_MAX, model, abort_field, rid,
                    )
                    await current_upstream.aclose()
                    retry_req = client.build_request(request.method, target, content=retry_body, headers=fwd_headers)
                    current_upstream = await client.send(retry_req, stream=True)
                    active_streams.register(rid, current_upstream)
                    continue
                # Kein JSON-Body zum Modifizieren oder Neuversuche aufgebraucht -
                # wie bisher mit sichtbarer Notiz endgültig abbrechen.
                yield _build_loop_abort_chunk(model, abort_field)
                status = "aborted_loop"
                logger.warning(
                    "Wiederholungsschleife erkannt und endgültig abgebrochen (nach %d Neuversuch(en)): "
                    "Modell '%s', Feld '%s' (rid=%s)",
                    attempt, model, abort_field, rid,
                )
                break
            if not is_stream and full_body:
                try:
                    obj = json.loads(bytes(full_body))
                    usage = obj.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    obj_choices = obj.get("choices") or []
                    if obj_choices:
                        finish_reason = obj_choices[0].get("finish_reason") or finish_reason
                        msg = obj_choices[0].get("message")
                        if msg is not None:
                            output_content_parts.append(msg.get("content") or "")
                            output_reasoning_parts.append(msg.get("reasoning") or msg.get("reasoning_content") or "")
                        elif obj_choices[0].get("text"):
                            output_content_parts.append(obj_choices[0]["text"])
                except Exception:
                    pass
        except Exception:
            # Von uns selbst ausgelöster Abbruch (siehe /requests/{rid}/cancel
            # unten) schließt upstream während dieser Schleife noch läuft - das
            # wirft hier eine ganz normale httpx-Exception, ist aber kein
            # Fehler, sondern das gewünschte Ergebnis. Alles andere (echter
            # Verbindungsabbruch zur Engine etc.) bleibt wie bisher "error"
            # inkl. Log/re-raise.
            if active_streams.was_cancelled(rid):
                status = "cancelled"
            else:
                status = "error"
                raise
        finally:
            disconnect_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_watcher
            active_streams.unregister(rid)
            _finish_and_record(
                rid, status, prompt_tokens, completion_tokens,
                request_messages=request_messages,
                request_prompt=request_prompt,
                request_params=request_params,
                output_content="".join(output_content_parts),
                output_reasoning="".join(output_reasoning_parts),
                finish_reason=finish_reason,
            )
            request_queue.release()
            await current_upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
