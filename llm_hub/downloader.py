"""HuggingFace-Modell-Downloads im Hintergrund, mit Fortschritts-Telemetrie
(Bytes, Prozent, Geschwindigkeit, ETA) zum Abfragen über HTTP/MCP.

Der eigentliche Transfer läuft in einem eigenen Kindprozess (siehe
download_worker.py), nicht in einem Thread - nur so lässt sich ein laufender
Download sauber abbrechen (POST .../cancel): huggingface_hubs
snapshot_download() hat keinen Abbruch-Mechanismus, und ein bereits laufender
Thread-Pool-Task kann aus Python heraus nicht mitten in der Ausführung
gestoppt werden (nur vor dem Start)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from . import catalog, config_editor
from .config import CONFIG_PATH, get_config

logger = logging.getLogger("llm_hub.downloader")

JOBS: dict[str, dict] = {}
# Prozess-Handles bewusst NICHT im JOBS-Dict selbst (das wird 1:1 als JSON für
# /dashboard/status, /models/pull, WS-Push ausgeliefert - ein
# asyncio.subprocess.Process-Objekt ist nicht serialisierbar).
_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
# hf_token ebenfalls bewusst NICHT im JOBS-Dict (siehe _PROCESSES-Kommentar
# oben) - JOBS wird 1:1 als JSON an GET /models/pull(/{job_id}) ausgeliefert,
# ein Token dort wäre für jeden Dashboard-Nutzer sichtbar.
_HF_TOKENS: dict[str, Optional[str]] = {}
POLL_INTERVAL = 2.0

# Persistiert laufende Downloads auf Platte (siehe _sync_pending_file /
# resume_pending_downloads) - JOBS ist nur In-Memory, ein 'systemctl restart'
# (oder Absturz) mitten in einem großen Download verlor den Job bisher
# komplett, obwohl huggingface_hub die .incomplete-Blobs liegen lässt und
# eigentlich nahtlos fortsetzen könnte (siehe _dir_size-Docstring oben). Live
# beobachtet: 2026-08-26, ein Neustart für ein anderes Feature brach nebenbei
# einen laufenden Download ab. Gleiches Muster wie process_manager.py
# LAST_ACTIVE_PATH (atomarer Schreib-über-.tmp-und-rename, gitignored - siehe
# .gitignore, enthält evtl. einen hf_token wie config.json auch).
PENDING_DOWNLOADS_PATH = CONFIG_PATH.parent / "pending_downloads.json"


def _sync_pending_file() -> None:
    """Schreibt den kompletten aktuellen Stand aller noch laufenden (nicht
    abgeschlossenen) Download-Jobs nach PENDING_DOWNLOADS_PATH - einfacher und
    weniger fehleranfällig als einzelne Einträge gezielt hinzuzufügen/zu
    entfernen. Wird bei jedem relevanten Zustandswechsel aufgerufen (siehe
    Aufrufer in _run_job/cancel_job unten)."""
    pending = [
        {"model": j["model"], "revision": j["revision"], "hf_token": _HF_TOKENS.get(j["job_id"])}
        for j in JOBS.values()
        if j["state"] in ("queued", "resolving", "downloading")
    ]
    try:
        tmp = PENDING_DOWNLOADS_PATH.with_suffix(PENDING_DOWNLOADS_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pending, f)
        tmp.replace(PENDING_DOWNLOADS_PATH)
    except Exception:
        logger.exception("Konnte pending_downloads.json nicht schreiben")


async def resume_pending_downloads() -> list[str]:
    """Beim Start aufgerufen (siehe main.py lifespan, analog zu
    process_manager.reap_orphan_engines) - setzt jeden beim letzten Beenden
    noch laufenden Download als neuen Job fort. huggingface_hub erkennt die
    liegen gebliebenen .incomplete-Blobs automatisch und lädt nur den Rest
    nach, kein Neustart bei Null. Gibt die Liste der wieder angestoßenen
    Modelle zurück (fürs Start-Log)."""
    if not PENDING_DOWNLOADS_PATH.exists():
        return []
    try:
        with open(PENDING_DOWNLOADS_PATH, encoding="utf-8") as f:
            pending = json.load(f)
    except Exception:
        logger.exception("pending_downloads.json ist beschädigt, ignoriere")
        return []
    resumed = []
    for entry in pending:
        model = entry.get("model")
        if not model:
            continue
        try:
            await start_job(model, entry.get("revision"), entry.get("hf_token"))
            resumed.append(model)
        except Exception:
            logger.exception("Konnte unterbrochenen Download für '%s' nicht fortsetzen", model)
    # start_job() selbst ruft _sync_pending_file() schon für jeden neu
    # gestarteten Job auf - die Datei ist an dieser Stelle also längst wieder
    # aktuell, kein zusätzliches Schreiben hier nötig.
    return resumed


def _cache_dir_for(model: str, hf_home: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + model.replace("/", "--"))


def _dir_size(path: Path) -> int:
    """Summiert die Bytes im blobs/-Verzeichnis eines Modells.

    Wichtig: huggingface_hub legt laufende Downloads als
    "<ziel-hash>.<zufalls-suffix>.incomplete" an. Wird der Manager-Prozess
    mitten im Download beendet (z.B. durch 'systemctl restart'), bleibt diese
    Datei als Leiche liegen; ein neuer Download-Versuch legt für denselben
    Ziel-Hash eine WEITERE .incomplete-Datei an - und setzt dabei NICHT etwa
    die alte fort, sondern beginnt bei 0 neu (live beobachtet: 2026-08-26,
    huggingface_hub 0.36 unter dieser Cache-Struktur; das widerspricht der
    Dokumentation, ist aber der real beobachtete Effekt). Ohne Deduplizierung
    würden alle Leichen mitgezählt und bytes_done weit über die echte
    Modellgröße hinausschießen (beobachtet: 39GB "done" bei 31GB Gesamtgröße).
    Pro Ziel-Hash zählt daher nur die größte (=am weitesten fortgeschrittene)
    .incomplete-Datei; fertige Blobs (ohne .incomplete-Suffix) sind durch
    huggingface_hub bereits pro Datei eindeutig und werden normal summiert.

    Zweite Falle, ebenfalls am 2026-08-26 live beobachtet (percent lief auf
    137% statt bei 100% stehenzubleiben): sobald der NEUE .incomplete-Download
    für einen Hash fertig ist, benennt huggingface_hub ihn zum finalen Blob
    (ohne .incomplete-Suffix) um - die ALTE Leiche für denselben Hash bleibt
    aber liegen und wurde vorher zusätzlich zum jetzt fertigen Blob gezählt
    (einmal via `completed`, einmal via `best_incomplete`). Deshalb zwei
    Durchgänge: erst alle fertigen Hashes sammeln, dann Leichen mit
    identischem Hash aus der Summe ausschließen."""
    if not path.exists():
        return 0
    completed = 0
    completed_hashes: set[str] = set()
    best_incomplete: dict[str, int] = {}
    entries: list[tuple[str, int]] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            entries.append((name, size))
            if not name.endswith(".incomplete"):
                completed += size
                completed_hashes.add(name)
    for name, size in entries:
        if name.endswith(".incomplete"):
            target_hash = name.split(".", 1)[0]
            if target_hash in completed_hashes:
                continue  # Leiche eines längst fertigen Blobs - siehe Docstring oben
            if size > best_incomplete.get(target_hash, -1):
                best_incomplete[target_hash] = size
    return completed + sum(best_incomplete.values())


async def start_job(model: str, revision: Optional[str] = None, hf_token: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "model": model,
        "revision": revision,
        "state": "queued",  # queued -> resolving -> downloading -> done/error/cancelled
        "bytes_total": 0,
        "bytes_done": 0,
        "percent": 0.0,
        "speed_mbps": 0.0,
        "eta_seconds": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    JOBS[job_id] = job
    _HF_TOKENS[job_id] = hf_token
    _sync_pending_file()
    asyncio.create_task(_run_job(job, hf_token))
    return job_id


async def _drain_worker_output(proc: asyncio.subprocess.Process, tail: list[str], max_lines: int = 40) -> None:
    """Liest laufend aus dem stdout/stderr-Pipe des Download-Kindprozesses
    (download_worker.py, stderr ist per stderr=STDOUT mit hineingemerged) -
    MUSS aktiv gelesen werden: der OS-Pipe-Puffer ist auf ~64KB begrenzt,
    ungelesen würde der Kindprozess irgendwann am nächsten Schreibversuch
    (z.B. eine weitere tqdm-Fortschrittszeile) hängen bleiben, sobald genug
    Ausgabe aufgelaufen ist. Hält nur die letzten `max_lines` Zeilen vor - für
    eine aussagekräftige Fehlermeldung bei Exit-Code != 0 (siehe _run_job)
    reicht das; ein voller Mitschnitt lohnt sich hier nicht (anders als beim
    NVFP4-Quantisierungs-Log, das der Nutzer live mitverfolgt)."""
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            return
        text = line.decode(errors="replace").rstrip("\n")
        if text:
            tail.append(text)
            if len(tail) > max_lines:
                del tail[0]


async def _run_worker_once(
    job: dict, worker_args: list[str], env: dict, blobs_dir: Path,
) -> tuple[int, list[str]]:
    """Ein einzelner Durchlauf des Download-Kindprozesses inkl. Fortschritts-
    Tracking (siehe _run_job für den Aufrufer, der das bei Bedarf einmal mit
    anderer env nachversucht - siehe dort). Gibt (Exit-Code, letzte
    Ausgabezeilen) zurück; ein Fehlschlag schon beim Start selbst wird als
    (1, [Fehlertext]) codiert, damit der Aufrufer beide Fälle einheitlich
    behandeln kann."""
    job_id = job["job_id"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *worker_args, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as e:
        logger.exception("Konnte Download-Prozess für '%s' nicht starten", job["model"])
        return 1, [f"Download-Prozess konnte nicht gestartet werden: {e}"]
    _PROCESSES[job_id] = proc

    tail: list[str] = []
    reader_task = asyncio.create_task(_drain_worker_output(proc, tail))
    last_bytes, last_t = 0, time.time()
    try:
        while proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass
            if job["state"] == "cancelled":
                break
            # _dir_size() ist blockierende Disk-I/O (os.walk + stat je Datei) - im
            # Thread ausführen, sonst friert das der Event-Loop für alle anderen
            # Requests/den Dashboard-Heartbeat kurz ein, siehe catalog.py-Docstring.
            now_bytes = await asyncio.to_thread(_dir_size, blobs_dir)
            now_t = time.time()
            dt = now_t - last_t
            if dt > 0:
                speed = max(now_bytes - last_bytes, 0) / dt
                job["speed_mbps"] = round(speed / 1_000_000, 2)
                if job["bytes_total"]:
                    remaining = max(job["bytes_total"] - now_bytes, 0)
                    job["eta_seconds"] = int(remaining / speed) if speed > 1e-6 else None
            job["bytes_done"] = now_bytes
            if job["bytes_total"]:
                job["percent"] = round(min(now_bytes / job["bytes_total"] * 100, 100), 1)
            last_bytes, last_t = now_bytes, now_t
    finally:
        _PROCESSES.pop(job_id, None)
        # Verteidigung gegen eine schmale Race Condition: cancel_job() greift
        # auf _PROCESSES zu, um den Prozess zu beenden - wurde der Job aber
        # abgebrochen, BEVOR der Prozess hier oben registriert war (zwischen
        # create_subprocess_exec() und der Zeile darunter), hätte cancel_job()
        # nichts zum Beenden gefunden. Hier deshalb sicherheitshalber nochmal
        # terminieren, falls der Prozess trotz "cancelled"-Status noch läuft.
        if job["state"] == "cancelled" and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        # Pipe bis EOF leerlesen (Prozess ist an dieser Stelle beendet oder
        # gerade eben terminiert worden) - sonst bliebe reader_task als
        # "pending task destroyed"-Warnung hängen.
        await reader_task

    return proc.returncode, tail


async def _run_job(job: dict, hf_token: Optional[str]) -> None:
    cfg = get_config()
    model = job["model"]
    job_id = job["job_id"]
    loop = asyncio.get_running_loop()
    job["state"] = "resolving"
    try:
        api = HfApi()
        info = await loop.run_in_executor(
            None,
            lambda: api.model_info(model, revision=job["revision"], files_metadata=True, token=hf_token),
        )
        job["bytes_total"] = sum((s.size or 0) for s in (info.siblings or []))
    except Exception as e:
        logger.exception("Konnte Metadaten für %s nicht laden", model)
        job["state"] = "error"
        job["error"] = f"Konnte Modell-Metadaten nicht laden: {e}"
        job["finished_at"] = time.time()
        _HF_TOKENS.pop(job_id, None)
        _sync_pending_file()
        return

    if job["state"] == "cancelled":  # zwischen Start und hier schon abgebrochen
        _HF_TOKENS.pop(job_id, None)
        _sync_pending_file()
        return

    job["state"] = "downloading"
    _sync_pending_file()
    blobs_dir = _cache_dir_for(model, cfg.resolved_hf_home()) / "blobs"

    worker_args = [sys.executable, "-m", "llm_hub.download_worker", model, "--cache-dir", str(Path(cfg.resolved_hf_home()) / "hub")]
    if job["revision"]:
        worker_args += ["--revision", job["revision"]]
    env = dict(os.environ)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    returncode, tail = await _run_worker_once(job, worker_args, env, blobs_dir)

    # Bekannter, reproduzierbarer Fehler von huggingface_hubs "Xet"-Downloadbackend
    # (hf-xet - seit huggingface_hub >=0.26 für darauf umgestellte Repos der
    # Default-Downloadpfad statt klassischem HTTP/LFS): bricht manche Repos
    # komplett mit "Unable to parse string as hex hash value" ab, statt sauber
    # auf HTTP zurückzufallen - live beobachtet 2026-08-28 bei
    # orcarouter/Qwen3.8-27B-Uncensored-FP8 (siehe Chat), reproduzierbar über
    # mehrere Versuche, aber verschwunden mit HF_HUB_DISABLE_XET=1 (klassischer
    # Pfad). HF_HUB_DISABLE_XET liest huggingface_hub NUR beim Modul-Import
    # (huggingface_hub.constants), ein In-Prozess-Retry mit geändertem
    # os.environ innerhalb desselben download_worker-Prozesses würde deshalb
    # NICHTS bewirken - der Retry hier startet bewusst einen ganz neuen
    # Kindprozess mit der Variable bereits VOR dessen Start gesetzt.
    if (
        returncode != 0
        and job["state"] != "cancelled"
        and "HF_HUB_DISABLE_XET" not in env
        and any("hex hash value" in line or "hf_xet" in line.lower() for line in tail)
    ):
        logger.warning(
            "Download von '%s' vermutlich am Xet-Backend gescheitert (%s) - "
            "versuche automatisch einmal mit HF_HUB_DISABLE_XET=1 erneut.",
            model, tail[-1] if tail else "?",
        )
        job["bytes_done"] = 0
        job["percent"] = 0.0
        retry_env = dict(env)
        retry_env["HF_HUB_DISABLE_XET"] = "1"
        returncode, tail = await _run_worker_once(job, worker_args, retry_env, blobs_dir)

    if job["state"] == "cancelled":
        # cancel_job() hat state/error/finished_at/Pending-Datei bereits
        # aktualisiert - hier nur noch aufräumen, nicht mit "done"/"error"
        # überschreiben.
        _HF_TOKENS.pop(job_id, None)
        return

    if returncode == 0:
        job["bytes_done"] = job["bytes_total"] or await asyncio.to_thread(_dir_size, blobs_dir)
        job["percent"] = 100.0
        job["state"] = "done"
        logger.info("Download abgeschlossen: %s", model)
        # Sofort sichtbar machen statt bis zu _CACHE_TTL Sekunden zu warten -
        # das neu heruntergeladene Modell soll im Dashboard/den APIs direkt
        # als "gecacht" auftauchen, mit korrekter (nicht 300s alter) Größe.
        catalog.invalidate_cache(cfg.resolved_hf_home())
        catalog.invalidate_size_cache(model)
        # Ohne das bliebe ein per pull_model()/POST /models/pull heruntergeladenes
        # Modell nur "gecacht, aber nicht registriert" (siehe GET /models) - lädt-
        # bar, aber nie in der eigentlichen config.json-Modellliste sichtbar,
        # bis es jemand manuell im Config-Editor nachträgt. Fire-and-forget: darf
        # den Download-Job selbst nicht mehr beeinflussen (ist ja schon "done").
        asyncio.create_task(config_editor.register_model_if_missing(
            model,
            note="Automatisch registriert nach Download (über /models/pull bzw. das "
                 "MCP-Tool pull_model). Bitte Werte prüfen (nur automatisch erkannt) - "
                 "gpu_memory_utilization ist ein konservativer Minimalwert zum sicheren "
                 "Start, für mehr Kontext/Durchsatz ggf. erhöhen.",
        ))
    else:
        logger.error("Download fehlgeschlagen: %s (exit code %s): %s", model, returncode, tail[-5:])
        job["state"] = "error"
        detail = "\n".join(tail[-5:])
        job["error"] = f"Download-Prozess beendet mit Exit-Code {returncode}." + (f" Letzte Ausgabe:\n{detail}" if detail else "")
    job["finished_at"] = time.time()
    _HF_TOKENS.pop(job_id, None)
    _sync_pending_file()


async def cancel_job(job_id: str) -> bool:
    """Bricht einen laufenden Download-Job ab (Button im Dashboard). Beendet
    den Kindprozess per SIGTERM (nach 5s Frist SIGKILL) - die bereits
    heruntergeladenen Teil-Dateien bleiben liegen (huggingface_hub kann einen
    Download bei erneutem Start fortsetzen; wer den Platz sofort zurückwill,
    nutzt den separaten "Von Platte löschen"-Button, siehe catalog.py). Gibt
    False zurück, wenn der Job unbekannt ist oder schon beendet war."""
    job = JOBS.get(job_id)
    if job is None or job["state"] not in ("queued", "resolving", "downloading"):
        return False
    job["state"] = "cancelled"
    job["error"] = "Vom Benutzer abgebrochen."
    job["finished_at"] = time.time()
    _sync_pending_file()
    proc = _PROCESSES.get(job_id)
    if proc is not None and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    return True


def list_jobs() -> list[dict]:
    return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)


def get_job(job_id: str) -> Optional[dict]:
    return JOBS.get(job_id)
