"""Automatisierter Performance-Tuner für ein einzelnes Modell (Dashboard-Button
"🚀 Auto-tune performance" im Config-Editor, siehe config_dashboard.py).

Hintergrund: 2026-08-25 wurden für Qwen3.8-27B-AWQ-INT4-v2 von Hand drei Hebel
live durchgemessen (siehe docs/benchmark-ollama-vs-vllm-qwen3.8-27b.md,
Abschnitt "Nachtrag") - cudagraph_capture_sizes (reiner Gewinn) und n-gram-
Spekulation (großer Gewinn bei Wiederholungen, kleiner Verlust bei freier
Prosa) wurden übernommen, FP8-KV-Cache verworfen (kein Nutzen bei kurzem
Kontext). Dieses Modul automatisiert GENAU diese von-Hand-Messung für ein
beliebiges Modell, damit sie nicht jedes Mal manuell per curl wiederholt
werden muss.

WICHTIG - das hier ist KEIN schneller Vorgang: jeder Test braucht einen
kompletten Kaltstart des Modells (bei einem 27B-Modell auf dieser Hardware
~200-300s), macht Baseline + 2 Varianten + finale Wiederherstellung also
4 volle Kaltstarts (~15-20 Minuten Gesamtlaufzeit). Läuft deshalb als
Hintergrund-Job (Polling wie downloader.py), NICHT synchron im Request.

Sicherheit:
- Nur EIN Tuning-Job gleichzeitig systemweit (GPU-exklusiv, siehe _current_job_id).
- Schreibt config.json während des Tests testweise um (cudagraph_capture_sizes/
  extra_args), stellt den ORIGINALEN Stand am Ende IMMER wieder her (auch bei
  Fehler/Abbruch, siehe finally-Block) und lädt das Modell danach ein letztes
  Mal mit dem wiederhergestellten Stand neu, damit der laufende Zustand nie
  von config.json abweicht.
- Ergebnisse werden NIE automatisch dauerhaft übernommen - der Job berichtet
  nur (wie capability_detector "Auto-detect"), der Nutzer entscheidet im
  Dashboard per "Übernehmen"-Button + dem normalen Save."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import httpx

from . import config_editor, process_manager, telemetry
from .config import get_config

logger = logging.getLogger("llm_hub.perf_tuner")

JOBS: dict[str, dict] = {}
_current_job_id: Optional[str] = None  # nur EIN Tuning-Job gleichzeitig, siehe Moduldocstring

# Feste Kandidaten - dieselben Werte, die sich am 2026-08-25 manuell bewährt
# haben (siehe Moduldocstring). Bewusst nicht pro Modell "optimiert" (z.B.
# num_speculative_tokens ans Modell anpassen) - das wäre ein eigenes,
# unabhängiges Tuning-Problem, hier geht es nur darum, OB die beiden Hebel
# für dieses Modell überhaupt etwas bringen.
_CUDAGRAPH_CANDIDATE = "1,2,4,8,16"
_SPECULATIVE_CONFIG_JSON = (
    '{"method": "ngram", "num_speculative_tokens": 5, '
    '"prompt_lookup_max": 4, "prompt_lookup_min": 2}'
)

# Zwei Test-Prompts mit bewusst gegensätzlichem Charakter (siehe Nachtrag im
# Benchmark-Dokument): freie Kreativantwort (wenig Wiederholung, worst case
# für n-gram-Spekulation) vs. wörtliches Kopieren (viel Wiederholung, best
# case). Ergebnis bei beiden zeigen, statt nur einer Zahl - genau die Nuance,
# die beim manuellen Test den Ausschlag gab.
_CREATIVE_PROMPT = "Erklaere in ca. 300 Woertern, wie Photosynthese auf molekularer Ebene funktioniert."
_REPETITION_SOURCE_TEXT = (
    "Die Photosynthese ist der fundamentale biochemische Prozess, durch den Pflanzen, "
    "Algen und bestimmte Bakterien Lichtenergie in chemisches Energiepotential umwandeln. "
    "Dieser Prozess laeuft in den Chloroplasten ab, genauer gesagt in den "
    "Thylakoidmembranen, wo Chlorophyll und andere Pigmente Lichtenergie absorbieren. "
    "Die Photosynthese gliedert sich in zwei Hauptphasen: die lichtabhaengigen "
    "Reaktionen und den Calvin-Zyklus."
)
_REPETITION_PROMPT = (
    "Gib mir bitte GENAU folgenden Text nochmal woertlich zurueck, ohne irgendetwas zu aendern:\n\n"
    + _REPETITION_SOURCE_TEXT
)


def _is_busy(model: str) -> bool:
    return any(r["model"] == model for r in telemetry.active_requests.values())


async def start_job(model: str) -> str:
    """Startet den Tuning-Job (Dashboard-Button). Wirft ValueError bei
    Vorbedingungen, die SOFORT prüfbar sind (unbekanntes/deaktiviertes Modell,
    bereits eine aktive Anfrage, oder schon ein anderer Tuning-Job aktiv) -
    alles andere (z.B. Kaltstart schlägt fehl) passiert erst im Hintergrund-Task
    und landet im Job-Status."""
    global _current_job_id
    if _current_job_id is not None and JOBS.get(_current_job_id, {}).get("state") == "running":
        raise ValueError(
            f"Es läuft bereits ein Performance-Test (für '{JOBS[_current_job_id]['model']}') - "
            f"immer nur einer gleichzeitig, da er die GPU exklusiv braucht."
        )
    cfg = get_config()
    ok, reason = process_manager.is_model_enabled(cfg, model)
    if not ok:
        raise ValueError(reason)
    if _is_busy(model):
        raise ValueError(
            f"'{model}' hat gerade eine aktive Anfrage - Performance-Test würde sie mitten "
            f"drin abbrechen (jeder Testschritt lädt die Engine neu). Bitte warten, bis sie fertig ist."
        )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "model": model,
        "state": "running",  # running -> done/error/cancelled
        "step": "queued",
        "findings": [],
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    JOBS[job_id] = job
    _current_job_id = job_id
    asyncio.create_task(_run_job(job))
    return job_id


async def cancel_job(job_id: str) -> bool:
    """Kooperativer Abbruch: wird zwischen den Testschritten geprüft (nicht
    mitten in einem laufenden Kaltstart/einer Generierung - das würde einen
    inkonsistenten Zwischenzustand riskieren). Der bereits laufende Schritt
    wird also noch zu Ende gebracht, danach überspringt der Job den Rest und
    stellt sofort den Originalzustand wieder her."""
    job = JOBS.get(job_id)
    if job is None or job["state"] != "running":
        return False
    job["_cancel_requested"] = True
    return True


def get_job(job_id: str) -> Optional[dict]:
    return JOBS.get(job_id)


def get_current_job() -> Optional[dict]:
    """Für die GPU-exklusiv-Prüfung ANDERER Hintergrund-Jobsysteme (siehe
    main.py: quantize_nvfp4_endpoint) - liefert den gerade laufenden Job
    dieses Moduls, falls einer läuft, sonst None. Fragt bewusst den
    tatsächlichen JOBS-Zustand ab statt ein zweites, separat gepflegtes Flag
    zu halten - sonst müsste jede der vielen return-Stellen in _run_job/
    cancel_job zusätzlich synchron gehalten werden (genau die Art Bug, die
    hier vermieden werden soll)."""
    if _current_job_id is not None:
        job = JOBS.get(_current_job_id)
        if job is not None and job["state"] == "running":
            return job
    return None


def list_jobs() -> list[dict]:
    return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)


async def _fetch_raw_metrics(engine_host: str, port: int) -> dict[str, float]:
    """Liest vLLMs /metrics roh (bewusst NICHT telemetry.fetch_engine_metrics()
    - dessen Cache würde einen frischen Wert direkt nach der Testanfrage evtl.
    noch als den alten (vor der Anfrage) zurückgeben)."""
    values: dict[str, float] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"http://{engine_host}:{port}/metrics")
        r.raise_for_status()
        text = r.text
    wanted = (
        "vllm:request_time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_count",
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for prefix in wanted:
            if line.startswith(prefix + "{"):
                try:
                    values[prefix] = values.get(prefix, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return values


async def _measure_decode(engine_host: str, port: int, model: str, prompt: str, max_tokens: int = 350) -> dict:
    """Eine einzelne Chat-Anfrage DIREKT an die Engine (nicht über den Manager-
    Proxy - RAG-Auto-Injection/Telemetrie würden die Zeitmessung verfälschen).
    Decode-Tok/s aus dem Delta von vLLMs eigener Prometheus-Metrik über genau
    diese eine Anfrage (dieselbe Methode wie beim manuellen Test 2026-08-25)."""
    before = await _fetch_raw_metrics(engine_host, port)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"http://{engine_host}:{port}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
    after = await _fetch_raw_metrics(engine_host, port)
    d_sum = after.get("vllm:request_time_per_output_token_seconds_sum", 0.0) - before.get(
        "vllm:request_time_per_output_token_seconds_sum", 0.0
    )
    d_cnt = after.get("vllm:request_time_per_output_token_seconds_count", 0.0) - before.get(
        "vllm:request_time_per_output_token_seconds_count", 0.0
    )
    tok_s = round(1.0 / (d_sum / d_cnt), 2) if d_cnt and d_sum > 0 else None
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {"tok_s": tok_s, "content": content}


async def _reload_and_time(model: str) -> float:
    """Modell hart entladen + über die normale Pool-Logik neu laden (reiht
    sich also ganz normal in eine evtl. laufende Warteschlange/Verdrängung
    ein, wie jeder andere Ladevorgang - siehe process_manager._make_room()).
    Gibt die Kaltstart-Dauer in Sekunden zurück."""
    await process_manager.stop_engine(model, reason="perf_tune")
    t0 = time.time()
    await process_manager.ensure_loaded(model, wait=True)
    return round(time.time() - t0, 1)


async def _run_job(job: dict) -> None:
    global _current_job_id
    model = job["model"]
    cfg = get_config()
    engine_host = cfg.engine_host

    mcfg_snapshot = cfg.models.get(model)
    original_cudagraph = mcfg_snapshot.cudagraph_capture_sizes if mcfg_snapshot else None
    original_extra_args = list(mcfg_snapshot.extra_args or []) if mcfg_snapshot else []
    already_has_speculative = "--speculative-config" in original_extra_args
    final_state = "error"  # sicherer Default, falls try/except unten aus irgendeinem Grund keinen erreicht

    def _cancelled() -> bool:
        return bool(job.get("_cancel_requested"))

    def _mutate_field(field: str, value):
        def _mutate(dump: dict):
            entry = dump["models"].setdefault(model, {})
            entry[field] = value
        return _mutate

    try:
        # --- Baseline: aktueller Stand, unverändert -------------------------
        job["step"] = "baseline"
        baseline_cold_start = await _reload_and_time(model)
        port = process_manager.engines[model].port
        baseline_creative = await _measure_decode(engine_host, port, model, _CREATIVE_PROMPT)
        baseline_repetition = await _measure_decode(engine_host, port, model, _REPETITION_PROMPT)

        # --- Test 1: cudagraph_capture_sizes --------------------------------
        if original_cudagraph:
            job["findings"].append({
                "test": "cudagraph_capture_sizes",
                "verdict": "already_configured",
                "current_value": original_cudagraph,
                "baseline_cold_start_s": baseline_cold_start,
            })
        elif _cancelled():
            pass
        else:
            job["step"] = "cudagraph_capture_sizes"
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("cudagraph_capture_sizes", _CUDAGRAPH_CANDIDATE))
            cold_start = await _reload_and_time(model)
            port = process_manager.engines[model].port
            decode = await _measure_decode(engine_host, port, model, _CREATIVE_PROMPT)
            improved_cold_start = cold_start < baseline_cold_start - 2  # etwas Toleranz gegen Messrauschen
            regressed_decode = (
                decode["tok_s"] is not None
                and baseline_creative["tok_s"] is not None
                and decode["tok_s"] < baseline_creative["tok_s"] * 0.9
            )
            job["findings"].append({
                "test": "cudagraph_capture_sizes",
                "verdict": "regressed" if regressed_decode else ("improved" if improved_cold_start else "no_change"),
                "recommended_value": _CUDAGRAPH_CANDIDATE,
                "baseline_cold_start_s": baseline_cold_start,
                "tuned_cold_start_s": cold_start,
                "baseline_decode_tok_s": baseline_creative["tok_s"],
                "tuned_decode_tok_s": decode["tok_s"],
            })
            # Zurücksetzen, damit der Speculative-Test unten wieder von einer
            # sauberen Baseline (nur EIN Hebel gleichzeitig verändert) ausgeht.
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("cudagraph_capture_sizes", original_cudagraph))

        # --- Test 2: n-gram-Spekulation --------------------------------------
        if already_has_speculative:
            job["findings"].append({
                "test": "speculative_decoding",
                "verdict": "already_configured",
            })
        elif _cancelled():
            pass
        else:
            job["step"] = "speculative_decoding"
            test_extra_args = original_extra_args + ["--speculative-config", _SPECULATIVE_CONFIG_JSON]
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("extra_args", test_extra_args))
            await _reload_and_time(model)  # Kaltstart-Zeit hier uninteressant, nur Decode zaehlt
            port = process_manager.engines[model].port
            creative = await _measure_decode(engine_host, port, model, _CREATIVE_PROMPT)
            repetition = await _measure_decode(engine_host, port, model, _REPETITION_PROMPT)
            correctness_ok = repetition["content"].strip() == _REPETITION_SOURCE_TEXT.strip()
            job["findings"].append({
                "test": "speculative_decoding",
                "verdict": "improved" if (
                    repetition["tok_s"] and baseline_repetition["tok_s"] and repetition["tok_s"] > baseline_repetition["tok_s"] * 1.1
                ) else "no_change",
                "recommended_extra_args": ["--speculative-config", _SPECULATIVE_CONFIG_JSON],
                "baseline_creative_tok_s": baseline_creative["tok_s"],
                "tuned_creative_tok_s": creative["tok_s"],
                "baseline_repetition_tok_s": baseline_repetition["tok_s"],
                "tuned_repetition_tok_s": repetition["tok_s"],
                "correctness_ok": correctness_ok,
            })
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("extra_args", original_extra_args))

        final_state = "cancelled" if _cancelled() else "done"
    except Exception as e:
        logger.exception("Performance-Test für '%s' fehlgeschlagen", model)
        final_state = "error"
        job["error"] = str(e)
    finally:
        # IMMER den Originalzustand wiederherstellen, auch bei Fehler/Abbruch -
        # config.json darf hier nie mit einem experimentellen Zwischenstand
        # zurückbleiben. WICHTIG: job["state"] bleibt bewusst bis HIERHER auf
        # "running" (nicht gleich zu Beginn des finally-Blocks auf final_state
        # gesetzt) - sonst könnte ein Poll genau in diesem Fenster "done"/
        # "error" sehen, obwohl der letzte Wiederherstellungs-Kaltstart (der
        # selbst mehrere Minuten dauert) noch gar nicht durch ist. Live
        # beobachtet: 2026-08-25, finished_at war noch null, state schon "done".
        job["step"] = "restoring"
        try:
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("cudagraph_capture_sizes", original_cudagraph))
            await asyncio.to_thread(config_editor.patch_config, _mutate_field("extra_args", original_extra_args))
            await _reload_and_time(model)
        except Exception:
            logger.exception(
                "Konnte Originalzustand für '%s' nach Performance-Test nicht sauber wiederherstellen - "
                "bitte Config-Editor/laufenden Zustand manuell prüfen.", model,
            )
        job["step"] = "done"
        job["state"] = final_state
        job["finished_at"] = time.time()
        if _current_job_id == job["job_id"]:
            _current_job_id = None
