"""Live System-Metriken (GPU/RAM) fürs Dashboard.

Die GB10 (Grace Blackwell) hat Unified Memory - RAM und GPU-Speicher teilen
sich denselben physischen Speicher. Deshalb liefert `nvidia-smi` hier für
memory.used/memory.total immer "Not Supported" (mit klassischem VRAM-Modell
nicht abbildbar). Die RAM-Auslastung aus /proc/meminfo deckt den tatsächlichen
Speicherbedarf (inkl. der vLLM-Modelle) daher schon vollständig ab.
Die GPU-Auslastung (Compute-%, Temperatur, Leistungsaufnahme) kommt weiterhin
aus nvidia-smi.
"""
from __future__ import annotations

import asyncio
import time

_CACHE_TTL = 0.8
_cache: dict | None = None
_cache_at: float = 0.0
_lock = asyncio.Lock()


def _num(value: str) -> float | None:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None


def _read_ram() -> dict:
    total_kb = avail_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                if total_kb is not None and avail_kb is not None:
                    break
    except OSError:
        total_kb = avail_kb = None
    if not total_kb or avail_kb is None:
        return {"ram_used_gb": None, "ram_free_gb": None, "ram_total_gb": None, "ram_percent": None}
    used_kb = total_kb - avail_kb
    return {
        "ram_used_gb": round(used_kb / 1_048_576, 2),
        # MemAvailable statt MemFree: berücksichtigt reclaimbaren Cache/Buffer,
        # ist also die realistische Zahl für "ohne Swappen tatsächlich noch
        # nutzbar" - genau das, was fürs Einschätzen von Kaltstarts zählt.
        "ram_free_gb": round(avail_kb / 1_048_576, 2),
        "ram_total_gb": round(total_kb / 1_048_576, 2),
        "ram_percent": round(used_kb / total_kb, 4),
    }


async def _read_gpu() -> dict:
    empty = {"gpu_percent": None, "gpu_power_w": None, "gpu_temp_c": None}
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
    except (OSError, asyncio.TimeoutError):
        return empty
    line = out.decode().strip().splitlines()[0] if out else ""
    parts = line.split(",")
    if len(parts) < 3:
        return empty
    util = _num(parts[0])
    return {
        "gpu_percent": (util / 100) if util is not None else None,
        "gpu_power_w": _num(parts[1]),
        "gpu_temp_c": _num(parts[2]),
    }


async def fetch_system_metrics() -> dict:
    """Kurz gecacht, damit mehrere offene Dashboard-Tabs nicht jede Sekunde
    mehrfach `nvidia-smi` als Subprozess starten."""
    global _cache, _cache_at
    async with _lock:
        now = time.time()
        if _cache is not None and (now - _cache_at) < _CACHE_TTL:
            return _cache
        gpu = await _read_gpu()
        ram = _read_ram()
        _cache = {**gpu, **ram}
        _cache_at = now
        return _cache
