"""Selbst-Quantisierung beliebiger HuggingFace-Modelle zu NVFP4 via NVIDIA
Model Optimizer, im Dashboard als Hintergrund-Job (siehe Chat vom
2026-08-26, Anleitung: https://build.nvidia.com/spark/nvfp4-quantization/instructions).

Anders als die schon in config.json registrierten NVFP4-Modelle (die als
FERTIG quantisierte Checkpoints von HuggingFace heruntergeladen wurden) läuft
die Quantisierung hier selbst, lokal, in einem NVIDIA-Docker-Container - für
Modelle, für die es (noch) keinen fertigen NVFP4-Upload gibt.

Ablauf (siehe _run_job): Docker-Container mit GPU-Zugriff starten, darin
NVIDIA/Model-Optimizer klonen+installieren, huggingface_example.sh mit
--quant nvfp4 laufen lassen (lädt das Original-Modell, kalibriert mit ein
paar hundert Beispiel-Prompts, exportiert NVFP4-Gewichte im normalen HF-
Safetensors-Format), Ergebnis nach models-quantized/<Name>-NVFP4/
verschieben und wie jedes andere lokale Modell automatisch in config.json
registrieren (config_editor.register_model_if_missing, dieselbe Funktion wie
bei einem fertigen Download - siehe downloader.py).

Container-Image/Model-Optimizer-Version sind auf die DGX-Spark-Variante der
Anleitung gepinnt (passt zur GB10-GPU dieser Maschine, siehe nvidia-smi) -
für andere Hardware (z.B. DGX Station/GB300) müssten Image und Optimizer-Tag
laut Anleitung angepasst werden."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from . import config_editor, process_manager
from .config import get_config

logger = logging.getLogger("vllm_manager.nvfp4_quantizer")

# Siehe Anleitung, Tabelle "Hardware platform" - DGX-Spark-Zeile (GB10, die
# GPU dieser Maschine). NICHT die DGX-Station/GB300-Variante (anderes Image,
# anderer Model-Optimizer-Tag, --gpus "device=$GPU_ID" statt "all").
DOCKER_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"
MODEL_OPTIMIZER_TAG = "0.35.0"

JOBS: dict[str, dict] = {}
_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_current_job_id: Optional[str] = None  # nur EIN Job gleichzeitig, siehe start_job()
MAX_LOG_LINES = 300

# Grobe, rein kosmetische Fortschrittsanzeige fürs Dashboard - ordnet neue
# Log-Zeilen anhand typischer Textmarker der Anleitung/des Skripts einer
# Phase zu (bewusst best-effort, kein echtes Protokoll des Skripts).
_STEP_MARKERS = [
    ("Cloning into", "cloning"),
    ("Installing collected packages", "installing"),
    ("Successfully installed", "installing"),
    ("Fetching", "downloading_model"),
    ("Downloading", "downloading_model"),
    ("calibrat", "quantizing"),  # "Calibrating"/"calibration" (Groß-/Kleinschreibung ignoriert, siehe unten)
    ("Quantiz", "quantizing"),
    ("Exporting", "exporting"),
    ("Export succeeded", "exporting"),
]


def _sanitize_name(hf_model_id: str) -> str:
    """'org/Model-Name' -> 'Model-Name', für den lokalen Zielordnernamen -
    dieselbe Konvention wie die bestehenden models-quantized/-Ordner
    (siehe config.json, z.B. Qwen3.8-27B-AWQ-INT4-v2)."""
    base = hf_model_id.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "-", base) or "model"


def _build_docker_cmd(
    hf_model_id: str, container_name: str, staging_dir: Path, hf_home: str, hf_token: Optional[str], tp: int,
    upgrade_transformers: bool = False,
) -> list[str]:
    # upgrade_transformers: Notausstieg für sehr neue Architekturen, die die im
    # Image gepinnte transformers-Version noch nicht kennt (live beobachtet
    # 2026-08-26: Qwen3.8-27B -> "KeyError: 'qwen3_5'", genau das von der
    # Fehlermeldung selbst vorgeschlagene 'pip install --upgrade transformers').
    # Standardmäßig AUS, weil das NVIDIAs sorgfältig aufeinander abgestimmte
    # Paketversionen (siehe die pip-Warnung "nvidia-modelopt ... incompatible"
    # weiter oben) für ALLE Modelle verändert, nicht nur für die betroffenen -
    # bei bereits unterstützten Architekturen also ein unnötiges Risiko.
    #
    # datasets läuft mit hoch (hat den zweiten Fehler unten NICHT behoben,
    # bleibt trotzdem drin - kann nicht schaden): das transformers-Upgrade
    # zieht transitiv ein neueres huggingface_hub mit (0.35.3 -> 1.28.0, ein
    # MAJOR-Sprung). In dessen 1.x-Zeile wurde die hf://-URI-Validierung
    # verschärft und lehnt jetzt "kanonische" Datasets ohne Namespace ab (z.B.
    # den Default-Kalibrierungsdatensatz "cnn_dailymail" - live beobachtet
    # 2026-08-26: "HfUriError: Repository id must be 'namespace/name', got
    # 'cnn_dailymail'"). Ein Downgrade von huggingface_hub geht nicht - neuere
    # transformers-Versionen verlangen zwingend huggingface-hub>=1.5.0.
    # Deshalb stattdessen ein Kalibrierungsdatensatz MIT Namespace (siehe
    # NVIDIA/Model-Optimizer modelopt/torch/utils/dataset_utils.py
    # SUPPORTED_DATASET_CONFIG - "magpie" -> Magpie-Align/Magpie-Pro-MT-300K-
    # v0.1, ein allgemeines Instruction-Tuning-Set, passender für ein
    # Chat-Modell als die code/math-spezifischen Alternativen dort).
    upgrade_step = "pip install --upgrade transformers datasets && " if upgrade_transformers else ""
    calib_dataset_args = " --calib_dataset magpie" if upgrade_transformers else ""
    # Der Container läuft als root (nötig fürs `pip install` in die System-
    # site-packages des Images) - Dateien in /workspace/output_models landen
    # dadurch root-owned, der Manager-Prozess (läuft als normaler Nutzer) kann
    # sie hinterher nicht verschieben/löschen (live beobachtet 2026-08-26:
    # "PermissionError: [Errno 13] Permission denied" beim shutil.move in
    # _run_job unten - Konsequenz aus genau der Warnung, die die NVIDIA-
    # Anleitung selbst im Cleanup-Abschnitt macht: "Quantization containers
    # may write ./output_models/ as root"). Deshalb am Ende (mit `;` statt
    # `&&`, damit es auch nach einem Fehlschlag noch versucht wird und
    # wenigstens das Aufräumen nicht zusätzlich an Rechten scheitert) auf den
    # Host-User zurück-chownen, den wir unten per -e an den Container geben.
    inner = (
        f"git clone -b {MODEL_OPTIMIZER_TAG} --single-branch "
        f"https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && "
        f"cd /app/Model-Optimizer && pip install -e '.[dev]' && "
        f"{upgrade_step}"
        f"export ROOT_SAVE_PATH=/workspace/output_models && "
        f"/app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh "
        f"--model '{hf_model_id}' --quant nvfp4 --tp {tp} --export_fmt hf{calib_dataset_args}"
        f"; chown -R $HOST_UID:$HOST_GID /workspace/output_models"
    )
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{staging_dir}:/workspace/output_models",
        # Eigenen hf_home mounten statt des Anleitungs-Defaults $HOME/.cache/
        # huggingface - dadurch nutzt der Container automatisch, was hier
        # schon gecacht ist (kein Doppel-Download), und alles landet im
        # selben Cache wie bei jedem regulären Modell-Download (siehe
        # downloader.py).
        "-v", f"{hf_home}:/root/.cache/huggingface",
        "-e", f"HOST_UID={os.getuid()}", "-e", f"HOST_GID={os.getgid()}",
    ]
    if hf_token:
        cmd += ["-e", f"HF_TOKEN={hf_token}"]
    cmd += [DOCKER_IMAGE, "bash", "-c", inner]
    return cmd


_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


async def start_job(
    hf_model_id: str, tp: int = 1, hf_token: Optional[str] = None, upgrade_transformers: bool = False,
) -> str:
    global _current_job_id
    # hf_model_id landet unten roh (in einfachen Anführungszeichen) in einem
    # verschachtelten `bash -c`-Kommando IM Container (siehe _build_docker_cmd)
    # - anders als beim äußeren docker-Aufruf (der geht über exec, kein Shell-
    # Parsing) wird dieser String dort tatsächlich von einer Shell gelesen.
    # Nur das übliche "org/name"-Format zulassen, alles andere (Anführungs-
    # zeichen, Semikolon, ...) ablehnen statt zu versuchen, es zu escapen.
    if not _MODEL_ID_RE.match(hf_model_id):
        raise ValueError(
            f"Ungültige Modell-ID '{hf_model_id}' - erwartet wird das übliche HuggingFace-Format "
            f"\"org/model-name\" (nur Buchstaben, Ziffern, '.', '_', '-')."
        )
    if _current_job_id is not None and JOBS.get(_current_job_id, {}).get("state") == "running":
        raise ValueError(
            f"Es läuft bereits eine NVFP4-Quantisierung (für '{JOBS[_current_job_id]['model']}') - "
            f"nur ein Job gleichzeitig (GPU-exklusiv)."
        )
    if process_manager.engines:
        loaded = ", ".join(sorted(process_manager.engines.keys()))
        raise ValueError(
            f"Bitte zuerst alle geladenen Modelle entladen ({loaded}) - die Quantisierung läuft in einem "
            f"separaten Docker-Container, der GPU-Speicher/Unified Memory unabhängig vom Hot Pool belegt "
            f"und sich mit laufenden Engines beißen würde."
        )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "model": hf_model_id,
        "tp": tp,
        "upgrade_transformers": upgrade_transformers,
        "state": "running",  # running -> done/error/cancelled
        "step": "starting",
        "log_tail": [],
        "registered_model": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    JOBS[job_id] = job
    _current_job_id = job_id
    asyncio.create_task(_run_job(job, hf_token))
    return job_id


def _append_log(job: dict, line: str) -> None:
    job["log_tail"].append(line)
    if len(job["log_tail"]) > MAX_LOG_LINES:
        del job["log_tail"][0]
    low = line.lower()
    for marker, step in _STEP_MARKERS:
        if marker.lower() in low:
            job["step"] = step
            break


async def _run_job(job: dict, hf_token: Optional[str]) -> None:
    global _current_job_id
    cfg = get_config()
    job_id = job["job_id"]
    hf_model_id = job["model"]
    container_name = f"vllm-manager-nvfp4-{job_id}"
    project_root = Path(cfg.hf_home).parent  # .../vllm (hf_home ist .../vllm/models)
    staging_dir = project_root / "quantize_output" / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = _build_docker_cmd(
            hf_model_id, container_name, staging_dir, cfg.hf_home, hf_token, job["tp"],
            job.get("upgrade_transformers", False),
        )
        _append_log(job, "$ " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as e:
        logger.exception("Konnte Quantisierungs-Container für '%s' nicht starten", hf_model_id)
        job["state"] = "error"
        job["error"] = f"Docker-Start fehlgeschlagen: {e}"
        job["finished_at"] = time.time()
        _current_job_id = None
        return
    _PROCESSES[job_id] = proc

    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            _append_log(job, line.decode(errors="replace").rstrip("\n"))
        returncode = await proc.wait()
    except Exception as e:
        logger.exception("Fehler beim Verfolgen des Quantisierungs-Containers für '%s'", hf_model_id)
        job["state"] = "error"
        job["error"] = f"Fehler während der Quantisierung: {e}"
        job["finished_at"] = time.time()
        _PROCESSES.pop(job_id, None)
        _current_job_id = None
        return
    finally:
        _PROCESSES.pop(job_id, None)

    if job["state"] == "cancelled":
        shutil.rmtree(staging_dir, ignore_errors=True)
        _current_job_id = None
        return

    if returncode != 0:
        job["state"] = "error"
        job["error"] = f"Container beendet mit Exit-Code {returncode} - siehe Log oben."
        job["finished_at"] = time.time()
        shutil.rmtree(staging_dir, ignore_errors=True)  # z.B. leerer/halber Output-Ordner - siehe Chat vom 2026-08-26
        _current_job_id = None
        return

    # Erwarteter Ordnername laut Anleitung: saved_models_<Modellname>_nvfp4_hf
    # - als Fallback (falls das Skript die Benennung mal ändert) nehmen wir
    # ersatzweise das einzige Unterverzeichnis, das tatsächlich entstanden
    # ist.
    produced = [d for d in staging_dir.iterdir() if d.is_dir()]
    if len(produced) == 1:
        produced_dir = produced[0]
    else:
        expected_name = f"saved_models_{_sanitize_name(hf_model_id)}_nvfp4_hf"
        expected = staging_dir / expected_name
        produced_dir = expected if expected.is_dir() else None
    if produced_dir is None:
        job["state"] = "error"
        job["error"] = (
            f"Container erfolgreich beendet, aber kein eindeutiges Ausgabeverzeichnis in {staging_dir} "
            f"gefunden (siehe Log) - manuell nachsehen."
        )
        job["finished_at"] = time.time()
        _current_job_id = None
        return

    dest = Path(project_root) / "models-quantized" / f"{_sanitize_name(hf_model_id)}-NVFP4"
    if dest.exists():
        dest = dest.with_name(dest.name + f"-{job_id[:6]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(produced_dir), str(dest))
    except Exception as e:
        logger.exception("Konnte Quantisierungs-Ergebnis für '%s' nicht nach %s verschieben", hf_model_id, dest)
        job["state"] = "error"
        job["error"] = f"Verschieben nach {dest} fehlgeschlagen: {e}"
        job["finished_at"] = time.time()
        _current_job_id = None
        return
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    model_path = str(dest)
    try:
        await config_editor.register_model_if_missing(
            model_path,
            note=(
                f"Selbst zu NVFP4 quantisiert aus '{hf_model_id}' via NVIDIA Model Optimizer "
                f"(Dashboard-Button \"NVFP4 quantisieren\", tp={job['tp']}). Bitte Werte prüfen (nur "
                f"automatisch erkannt) - gpu_memory_utilization ist ein konservativer Minimalwert."
            ),
        )
    except Exception:
        logger.exception("Konnte quantisiertes Modell '%s' nicht automatisch registrieren", model_path)
        # Datei liegt trotzdem an ihrem Platz - manuell im Config-Editor
        # nachtragbar, deshalb kein "error"-Status nur wegen der Registrierung.

    job["registered_model"] = model_path
    job["state"] = "done"
    job["finished_at"] = time.time()
    logger.info("NVFP4-Quantisierung abgeschlossen: '%s' -> %s", hf_model_id, model_path)
    _current_job_id = None


async def cancel_job(job_id: str) -> bool:
    """Bricht einen laufenden Quantisierungs-Job ab (Dashboard-Button). Nutzt
    zusätzlich zum SIGTERM auf den lokalen `docker run`-Client-Prozess auch
    ein explizites `docker stop` auf den benannten Container - `docker run`
    im Vordergrund reicht Signale zwar normalerweise durch, ein benannter
    Container lässt sich damit aber unabhängig davon zuverlässig beenden."""
    job = JOBS.get(job_id)
    if job is None or job["state"] != "running":
        return False
    job["state"] = "cancelled"
    job["error"] = "Vom Benutzer abgebrochen."
    job["finished_at"] = time.time()
    try:
        stop_proc = await asyncio.create_subprocess_exec(
            "docker", "stop", f"vllm-manager-nvfp4-{job_id}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(stop_proc.wait(), timeout=15)
    except Exception:
        pass  # Container evtl. schon weg - der proc.terminate()-Fallback unten deckt den Rest ab
    proc = _PROCESSES.get(job_id)
    if proc is not None and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    global _current_job_id
    if _current_job_id == job_id:
        _current_job_id = None
    return True


def list_jobs() -> list[dict]:
    return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)


def get_job(job_id: str) -> Optional[dict]:
    return JOBS.get(job_id)
