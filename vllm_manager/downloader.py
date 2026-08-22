"""HuggingFace-Modell-Downloads im Hintergrund, mit Fortschritts-Telemetrie
(Bytes, Prozent, Geschwindigkeit, ETA) zum Abfragen über HTTP/MCP."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from .config import get_config

logger = logging.getLogger("vllm_manager.downloader")

JOBS: dict[str, dict] = {}
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
        "state": "queued",  # queued -> resolving -> downloading -> done/error
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

    job["state"] = "downloading"
    blobs_dir = _cache_dir_for(model, cfg.hf_home) / "blobs"

    download_future = loop.run_in_executor(
        None,
        lambda: snapshot_download(
            repo_id=model,
            revision=job["revision"],
            token=hf_token,
            cache_dir=str(Path(cfg.hf_home) / "hub"),
        ),
    )

    last_bytes, last_t = 0, time.time()
    while not download_future.done():
        await asyncio.sleep(POLL_INTERVAL)
        now_bytes = _dir_size(blobs_dir)
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

    try:
        await download_future
        job["bytes_done"] = job["bytes_total"] or _dir_size(blobs_dir)
        job["percent"] = 100.0
        job["state"] = "done"
        logger.info("Download abgeschlossen: %s", model)
    except Exception as e:
        logger.exception("Download fehlgeschlagen: %s", model)
        job["state"] = "error"
        job["error"] = str(e)
    job["finished_at"] = time.time()


def list_jobs() -> list[dict]:
    return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)


def get_job(job_id: str) -> Optional[dict]:
    return JOBS.get(job_id)
