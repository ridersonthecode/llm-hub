"""Startet/stoppt die eigentliche vLLM-Engine (vllm serve <model>) als Kindprozess
bei Bedarf - ohne dass der vllm-Manager-Dienst selbst neu gestartet werden muss."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx

from .config import Config, get_config

logger = logging.getLogger("vllm_manager.engine")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 50
# Verlauf abgeschlossener Modell-Sessions (neueste zuerst) - wie "ollama ps",
# nur historisch statt nur der aktuelle Zustand. Überlebt keinen Dienst-Neustart.
model_history: deque[dict] = deque(maxlen=MAX_HISTORY)


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
    """Hält den Zustand des aktuell laufenden vLLM-Engine-Kindprozesses.
    Es läuft (aktuell) immer höchstens ein Modell gleichzeitig."""

    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.started_at: Optional[float] = None
        self.log_path: Optional[Path] = None
        self.last_used: float = time.time()
        self.lock = asyncio.Lock()
        # "idle" (nichts geladen) | "loading" (Kaltstart läuft) | "ready" (bereit für Requests)
        self.state: str = "idle"
        self.last_error: Optional[str] = None

    def status(self) -> dict:
        running = self.process is not None and self.process.returncode is None
        return {
            "loaded_model": self.model,
            "state": self.state,
            "running": running,
            "started_at": self.started_at,
            "uptime_seconds": (time.time() - self.started_at) if self.started_at and running else None,
            "log_file": str(self.log_path) if self.log_path else None,
            "last_error": self.last_error,
        }


engine = EngineState()


async def _health_ok(cfg: Config) -> bool:
    url = f"http://{cfg.engine_host}:{cfg.engine_port}/health"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(url)
            return r.status_code == 200
    except Exception:
        return False


def _build_command(cfg: Config, model: str) -> list[str]:
    mcfg = cfg.models.get(model)
    gmu, mml = cfg.serve_args_for(model)
    cmd = [
        cfg.resolved_vllm_bin(), "serve", model,
        "--host", cfg.engine_host,
        "--port", str(cfg.engine_port),
        "--gpu-memory-utilization", str(gmu),
        "--max-model-len", str(mml),
    ]
    if mcfg:
        if mcfg.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        if mcfg.tool_call_parser:
            cmd += ["--tool-call-parser", mcfg.tool_call_parser]
        cmd += list(mcfg.extra_args or [])
    return cmd


async def stop_engine(reason: str = "manual_unload", timeout: float = 30.0) -> None:
    """Beendet die laufende Engine (falls vorhanden) und schreibt einen
    Verlaufseintrag. `reason` z.B. "manual_unload", "idle_timeout", "crashed",
    "timeout", "shutdown" oder "replaced_by:<neues_modell>"."""
    proc = engine.process
    if engine.model is not None:
        model_history.appendleft({
            "model": engine.model,
            "loaded_at": engine.started_at,
            "unloaded_at": time.time(),
            "duration_seconds": round(time.time() - engine.started_at, 1) if engine.started_at else None,
            "reason": reason,
        })
    if proc is None:
        engine.model = None
        engine.state = "idle"
        return
    if proc.returncode is None:
        logger.info("Beende Engine-Prozess für Modell %s (pid=%s, Grund: %s)", engine.model, proc.pid, reason)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Engine reagiert nicht auf SIGTERM, sende SIGKILL")
            proc.kill()
            await proc.wait()
    engine.process = None
    engine.model = None
    engine.started_at = None
    engine.state = "idle"


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

    async with engine.lock:
        if (
            engine.model == model
            and engine.state == "ready"
            and engine.process is not None
            and engine.process.returncode is None
        ):
            engine.last_used = time.time()
            return engine.status()

        if engine.model is not None and engine.process is not None:
            logger.info("Wechsle Modell: %s -> %s (altes Modell wird zuerst entladen)", engine.model, model)
            await stop_engine(reason=f"replaced_by:{model}")

        engine.state = "loading"
        engine.last_error = None
        log_path = LOG_DIR / f"{_safe_name(model)}.log"
        engine.log_path = log_path
        cmd = _build_command(cfg, model)
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
        engine.process = proc
        engine.model = model
        engine.started_at = time.time()

        if not wait:
            return engine.status()

        deadline = time.time() + cfg.startup_timeout_seconds
        while time.time() < deadline:
            if engine.process.returncode is not None:
                tail = _tail(engine.log_path)
                msg = f"vLLM-Engine für '{model}' ist unerwartet beendet (exit={proc.returncode})."
                engine.last_error = msg
                await stop_engine(reason="crashed")
                raise RuntimeError(f"{msg}\nLetzte Log-Zeilen ({log_path}):\n{tail}")
            if await _health_ok(cfg):
                engine.state = "ready"
                engine.last_used = time.time()
                return engine.status()
            await asyncio.sleep(2)

        tail = _tail(engine.log_path)
        msg = f"Timeout beim Start von '{model}' nach {cfg.startup_timeout_seconds}s."
        engine.last_error = msg
        await stop_engine(reason="timeout")
        raise TimeoutError(f"{msg}\nLetzte Log-Zeilen ({log_path}):\n{tail}")
