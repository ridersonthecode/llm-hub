"""Startet/stoppt vLLM-Engine-Kindprozesse (vllm serve <model>) bei Bedarf -
ohne dass der vllm-Manager-Dienst selbst neu gestartet werden muss.

Hot Pool: bei max_concurrent_models > 1 können mehrere Modelle gleichzeitig
geladen bleiben (je ein Kindprozess auf einem eigenen Port aus dem Pool
engine_port, engine_port+1, ...). Ein Wechsel zwischen bereits geladenen
Modellen ist dann instant statt eines Kaltstarts. Reicht der Speicher (Summe
der gpu_memory_utilization aller Engines, siehe gpu_memory_ceiling) oder die
Poolgröße nicht, wird die am längsten ungenutzte Engine verdrängt (LRU) -
Engines mit gerade aktiven Anfragen werden dabei übersprungen."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx

from . import telemetry
from .config import CONFIG_PATH, Config, get_config

logger = logging.getLogger("vllm_manager.engine")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 50
# Verlauf abgeschlossener Modell-Sessions (neueste zuerst) - wie "ollama ps",
# nur historisch statt nur der aktuelle Zustand. Überlebt keinen Dienst-Neustart.
model_history: deque[dict] = deque(maxlen=MAX_HISTORY)

# Persistenter Zeiger aufs zuletzt genutzte Modell (nicht PROJECT_ROOT fest
# verdrahtet, sondern neben der tatsächlich verwendeten config.json - folgt
# damit VLLM_MANAGER_CONFIG, siehe config_editor.py fürs selbe Muster). Wird
# bei jeder Nutzung eines bereit stehenden Modells aktualisiert (Kaltstart
# fertig ODER Wiederverwendung eines schon warmen Modells) und beim
# automatischen Nachladen in main.py's lifespan() gelesen.
LAST_ACTIVE_PATH = CONFIG_PATH.parent / "last_active_model.json"
_last_persisted_model: Optional[str] = None


def _persist_last_active(model: str) -> None:
    global _last_persisted_model
    if model == _last_persisted_model:
        return  # unverändert - unnötigen Disk-I/O bei jeder Anfrage vermeiden
    try:
        tmp = LAST_ACTIVE_PATH.with_suffix(LAST_ACTIVE_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"model": model, "saved_at": time.time()}, f)
        tmp.replace(LAST_ACTIVE_PATH)
        _last_persisted_model = model
    except OSError:
        logger.exception("Konnte last_active_model.json nicht schreiben")


def _clear_last_active(model: str) -> None:
    """Nur aufrufen bei explizitem manuellem Entladen - ein automatisch
    verdrängtes/idle-getimeoutetes Modell soll nach einem Neustart trotzdem
    zurückkommen, ein bewusst vom Nutzer entladenes nicht."""
    global _last_persisted_model
    if load_last_active_model() != model:
        return
    _last_persisted_model = None
    try:
        LAST_ACTIVE_PATH.unlink(missing_ok=True)
    except OSError:
        logger.exception("Konnte last_active_model.json nicht löschen")


def load_last_active_model() -> Optional[str]:
    if not LAST_ACTIVE_PATH.exists():
        return None
    try:
        with open(LAST_ACTIVE_PATH, encoding="utf-8") as f:
            return json.load(f).get("model")
    except (OSError, json.JSONDecodeError):
        logger.exception("last_active_model.json ist beschädigt, ignoriere")
        return None


def _safe_name(model: str) -> str:
    return model.replace("/", "__")


def _tail(path: Optional[Path], n: int = 40) -> str:
    if not path or not path.exists():
        return ""
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


class EngineState:
    """Hält den Zustand eines einzelnen vLLM-Engine-Kindprozesses für EIN
    Modell. Mehrere Instanzen können gleichzeitig existieren (siehe `engines`
    unten), je nach max_concurrent_models."""

    def __init__(self, model: str, port: int) -> None:
        self.model = model
        self.port = port
        self.process: Optional[asyncio.subprocess.Process] = None
        self.started_at: Optional[float] = None
        self.log_path: Optional[Path] = None
        self.last_used: float = time.time()
        # "loading" (Kaltstart läuft) | "ready" (bereit für Requests)
        self.state: str = "loading"
        self.last_error: Optional[str] = None

    def status(self) -> dict:
        running = self.process is not None and self.process.returncode is None
        return {
            "loaded_model": self.model,
            "port": self.port,
            "state": self.state,
            "running": running,
            "started_at": self.started_at,
            "uptime_seconds": (time.time() - self.started_at) if self.started_at and running else None,
            "log_file": str(self.log_path) if self.log_path else None,
            "last_error": self.last_error,
        }


# model -> EngineState. Bei max_concurrent_models=1 (Default) hält dieses
# Dict wie zuvor höchstens einen Eintrag.
engines: dict[str, EngineState] = {}
_pool_lock = asyncio.Lock()


def loaded_models() -> list[str]:
    """Modelle, die aktuell bereit (nicht nur im Kaltstart) sind."""
    return [m for m, e in engines.items() if e.state == "ready"]


def is_ready(model: str) -> bool:
    e = engines.get(model)
    return e is not None and e.state == "ready" and e.process is not None and e.process.returncode is None


async def _health_ok(cfg: Config, port: int) -> bool:
    url = f"http://{cfg.engine_host}:{port}/health"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(url)
            return r.status_code == 200
    except Exception:
        return False


def _build_command(cfg: Config, model: str, port: int) -> list[str]:
    mcfg = cfg.models.get(model)
    gmu, mml = cfg.serve_args_for(model)
    cmd = [
        cfg.resolved_vllm_bin(), "serve", model,
        "--host", cfg.engine_host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gmu),
        "--max-model-len", str(mml),
    ]
    if mcfg:
        if mcfg.task == "embed":
            # vLLM >=0.26 hat das alte "--task" durch "--runner" ersetzt.
            cmd += ["--runner", "pooling"]
        if mcfg.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        if mcfg.tool_call_parser:
            cmd += ["--tool-call-parser", mcfg.tool_call_parser]
        if mcfg.reasoning_parser:
            cmd += ["--reasoning-parser", mcfg.reasoning_parser]
        cmd += list(mcfg.extra_args or [])
    return cmd


def _allocate_port(cfg: Config) -> int:
    used = {e.port for e in engines.values()}
    for i in range(cfg.max_concurrent_models):
        port = cfg.engine_port + i
        if port not in used:
            return port
    raise RuntimeError(
        "Kein freier Engine-Port im Pool - die Verdrängungslogik hätte vorher "
        "schon Platz schaffen müssen. Das ist ein Bug, bitte melden."
    )


async def _make_room(cfg: Config, model: str, gmu: float) -> None:
    """Verdrängt bei Bedarf Engines (LRU), damit `model` (mit geschätztem
    Speicherbedarf `gmu`) ins Poolgrößen- und Speicherbudget passt. Engines mit
    gerade aktiven Anfragen werden NIE automatisch verdrängt - das würde eine
    laufende Antwort mitten drin abbrechen. Reicht der Platz nur durch
    Verdrängen einer gerade beschäftigten Engine, schlägt das Laden stattdessen
    mit einer klaren Fehlermeldung fehl (propagiert als HTTP 503/500 zum
    Aufrufer, siehe main.py/ollama_compat.py) - der Nutzer muss dann bewusst
    manuell ein Modell entladen."""
    protected = {r["model"] for r in telemetry.active_requests.values()}

    def others() -> list[EngineState]:
        return [e for e in engines.values() if e.model != model]

    def total_util() -> float:
        return sum(cfg.serve_args_for(e.model)[0] for e in others()) + gmu

    while len(others()) >= cfg.max_concurrent_models or total_util() > cfg.gpu_memory_ceiling:
        pool = others()
        if not pool:
            # Nichts zum Verdrängen da, aber das Modell passt trotzdem nicht -
            # muss an gpu_memory_ceiling/gpu_memory_utilization liegen, nicht an
            # aktiven Anfragen.
            raise RuntimeError(
                f"Modell '{model}' passt nicht ins GPU-Speicherbudget "
                f"(gpu_memory_utilization={gmu} vs. gpu_memory_ceiling={cfg.gpu_memory_ceiling}), "
                f"auch mit leerem Hot Pool nicht. gpu_memory_ceiling oder die "
                f"gpu_memory_utilization dieses Modells in config.json anpassen."
            )
        candidates = [e for e in pool if e.model not in protected]
        if not candidates:
            busy = sorted({e.model for e in pool if e.model in protected})
            raise RuntimeError(
                f"Kein Platz im Hot Pool für '{model}': alle {len(pool)} aktuell "
                f"geladenen Modelle ({', '.join(busy)}) haben gerade eine laufende "
                f"Anfrage und werden nicht automatisch verdrängt, um sie nicht "
                f"abzubrechen. Bitte manuell ein Modell entladen (Dashboard-Button "
                f"oder POST /models/<model>/unload) und die Anfrage erneut senden."
            )
        victim = min(candidates, key=lambda e: e.last_used)
        logger.info("Verdränge Modell '%s' aus dem Pool, um Platz für '%s' zu schaffen", victim.model, model)
        await stop_engine(victim.model, reason=f"evicted_for:{model}")


async def stop_engine(model: Optional[str] = None, reason: str = "manual_unload", timeout: float = 30.0) -> None:
    """Beendet eine laufende Engine und schreibt einen Verlaufseintrag.
    Ohne `model` werden ALLE laufenden Engines beendet (z.B. beim Dienst-
    Shutdown). `reason` z.B. "manual_unload", "idle_timeout", "crashed",
    "timeout", "shutdown" oder "evicted_for:<neues_modell>"."""
    if model is None:
        for m in list(engines.keys()):
            await stop_engine(m, reason=reason, timeout=timeout)
        return

    eng = engines.get(model)
    if eng is None:
        return
    if reason == "manual_unload":
        _clear_last_active(model)
    if eng.started_at is not None:
        model_history.appendleft({
            "model": eng.model,
            "loaded_at": eng.started_at,
            "unloaded_at": time.time(),
            "duration_seconds": round(time.time() - eng.started_at, 1),
            "reason": reason,
            "error": eng.last_error if reason == "crashed" else None,
        })
    proc = eng.process
    if proc is not None and proc.returncode is None:
        logger.info("Beende Engine-Prozess für Modell %s (pid=%s, Port=%s, Grund: %s)", eng.model, proc.pid, eng.port, reason)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Engine reagiert nicht auf SIGTERM, sende SIGKILL")
            proc.kill()
            await proc.wait()
    engines.pop(model, None)


def is_model_enabled(cfg: Config, model: str) -> tuple[bool, Optional[str]]:
    mcfg = cfg.models.get(model)
    if mcfg is not None and not mcfg.enabled:
        return False, mcfg.notes or f"Modell '{model}' ist in config.json als enabled:false markiert."
    return True, None


async def ensure_loaded(model: str, wait: bool = True) -> dict:
    cfg = get_config()
    ok, reason = is_model_enabled(cfg, model)
    if not ok:
        raise RuntimeError(reason)

    async with _pool_lock:
        existing = engines.get(model)
        if existing is not None and is_ready(model):
            existing.last_used = time.time()
            _persist_last_active(model)
            return existing.status()
        if existing is not None:
            # Hängender/abgestürzter Eintrag - erst aufräumen, dann sauber neu starten.
            await stop_engine(model, reason="restart")

        gmu, mml = cfg.serve_args_for(model)
        await _make_room(cfg, model, gmu)
        port = _allocate_port(cfg)

        eng = EngineState(model, port)
        engines[model] = eng
        log_path = LOG_DIR / f"{_safe_name(model)}.log"
        eng.log_path = log_path
        cmd = _build_command(cfg, model, port)
        env = os.environ.copy()
        env["HF_HOME"] = cfg.hf_home
        # systemd setzt kein venv-PATH: Tools wie 'ninja' (von flashinfer beim
        # JIT-Kompilieren von Sampling-Kerneln benötigt) liegen in .venv/bin und
        # würden sonst nicht über PATH gefunden (-> Engine-Crash exit=1).
        venv_bin = str(Path(cfg.resolved_vllm_bin()).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        mcfg = cfg.models.get(model)
        if mcfg and mcfg.hf_token:
            env["HF_TOKEN"] = mcfg.hf_token
        logger.info("Starte Engine: %s", " ".join(cmd))
        log_f = open(log_path, "ab")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_f,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(cfg.hf_home).parent),
        )
        eng.process = proc
        eng.started_at = time.time()

        if not wait:
            return eng.status()

        deadline = time.time() + cfg.startup_timeout_seconds
        while time.time() < deadline:
            if eng.process.returncode is not None:
                tail = _tail(eng.log_path)
                msg = f"vLLM-Engine für '{model}' ist unerwartet beendet (exit={proc.returncode})."
                eng.last_error = msg
                await stop_engine(model, reason="crashed")
                raise RuntimeError(f"{msg}\nLetzte Log-Zeilen ({log_path}):\n{tail}")
            if await _health_ok(cfg, port):
                eng.state = "ready"
                eng.last_used = time.time()
                _persist_last_active(model)
                return eng.status()
            await asyncio.sleep(2)

        tail = _tail(eng.log_path)
        msg = f"Timeout beim Start von '{model}' nach {cfg.startup_timeout_seconds}s."
        eng.last_error = msg
        await stop_engine(model, reason="timeout")
        raise TimeoutError(f"{msg}\nLetzte Log-Zeilen ({log_path}):\n{tail}")
