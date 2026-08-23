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
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from . import catalog
from .config import get_config

logger = logging.getLogger("vllm_manager.downloader")

JOBS: dict[str, dict] = {}
# Prozess-Handles bewusst NICHT im JOBS-Dict selbst (das wird 1:1 als JSON für
# /dashboard/status, /models/pull, WS-Push ausgeliefert - ein
# asyncio.subprocess.Process-Objekt ist nicht serialisierbar).
_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
POLL_INTERVAL = 2.0


def _cache_dir_for(model: str, hf_home: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + model.replace("/", "--"))


def _dir_size(path: Path) -> int:
    """Summiert die Bytes im blobs/-Verzeichnis eines Modells.

    Wichtig: huggingface_hub legt laufende Downloads als
    "<ziel-hash>.<zufalls-suffix>.incomplete" an. Wird der Manager-Prozess
    mitten im Download beendet (z.B. durch 'systemctl restart'), bleibt diese
    Datei als Leiche liegen; ein neuer Download-Versuch legt für denselben
    Ziel-Hash eine WEITERE .incomplete-Datei an. Ohne Deduplizierung würden
    alle Leichen mitgezählt und bytes_done weit über die echte Modellgröße
    hinausschießen (beobachtet: 39GB "done" bei 31GB Gesamtgröße). Pro
    Ziel-Hash zählt daher nur die größte (=am weitesten fortgeschrittene)
    .incomplete-Datei; fertige Blobs (ohne .incomplete-Suffix) sind durch
    huggingface_hub bereits pro Datei eindeutig und werden normal summiert.
    """
    if not path.exists():
        return 0
    completed = 0
    best_incomplete: dict[str, int] = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if name.endswith(".incomplete"):
                target_hash = name.split(".", 1)[0]
                if size > best_incomplete.get(target_hash, -1):
                    best_incomplete[target_hash] = size
            else:
                completed += size
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
    asyncio.create_task(_run_job(job, hf_token))
    return job_id


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
        return

    if job["state"] == "cancelled":  # zwischen Start und hier schon abgebrochen
        return

    job["state"] = "downloading"
    blobs_dir = _cache_dir_for(model, cfg.hf_home) / "blobs"

    worker_args = [sys.executable, "-m", "vllm_manager.download_worker", model, "--cache-dir", str(Path(cfg.hf_home) / "hub")]
    if job["revision"]:
        worker_args += ["--revision", job["revision"]]
    env = dict(os.environ)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    try:
        proc = await asyncio.create_subprocess_exec(
            *worker_args, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as e:
        logger.exception("Konnte Download-Prozess für %s nicht starten", model)
        job["state"] = "error"
        job["error"] = f"Download-Prozess konnte nicht gestartet werden: {e}"
        job["finished_at"] = time.time()
        return
    _PROCESSES[job_id] = proc

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

    if job["state"] == "cancelled":
        # cancel_job() hat state/error/finished_at bereits gesetzt - hier nur
        # noch aufräumen, nicht mit "done"/"error" überschreiben.
        return

    if proc.returncode == 0:
        job["bytes_done"] = job["bytes_total"] or await asyncio.to_thread(_dir_size, blobs_dir)
        job["percent"] = 100.0
        job["state"] = "done"
        logger.info("Download abgeschlossen: %s", model)
        # Sofort sichtbar machen statt bis zu _CACHE_TTL Sekunden zu warten -
        # das neu heruntergeladene Modell soll im Dashboard/den APIs direkt
        # als "gecacht" auftauchen, mit korrekter (nicht 300s alter) Größe.
        catalog.invalidate_cache(cfg.hf_home)
        catalog.invalidate_size_cache(model)
    else:
        logger.error("Download fehlgeschlagen: %s (exit code %s)", model, proc.returncode)
        job["state"] = "error"
        job["error"] = f"Download-Prozess beendet mit Exit-Code {proc.returncode}."
    job["finished_at"] = time.time()


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
