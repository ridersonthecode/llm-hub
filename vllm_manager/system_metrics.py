"""Live System-Metriken (GPU/RAM/CPU) fürs Dashboard.

Die GB10 (Grace Blackwell) hat Unified Memory - RAM und GPU-Speicher teilen
sich denselben physischen Speicher. Deshalb liefert `nvidia-smi` hier für
memory.used/memory.total immer "Not Supported" (mit klassischem VRAM-Modell
nicht abbildbar). Die RAM-Auslastung aus /proc/meminfo deckt den tatsächlichen
Speicherbedarf (inkl. der vLLM-Modelle) daher schon vollständig ab.
Die GPU-Auslastung (Compute-%, Temperatur, Leistungsaufnahme) kommt weiterhin
aus nvidia-smi. Die CPU-Auslastung (Gesamt + pro Kern, siehe _read_cpu())
kommt aus /proc/stat - auf der 20-Kern-Grace-CPU dieses Systems relevant,
weil einzelne Anfragen (Tokenizer, Sampling, RAG-Chunking) durchaus einzelne
Kerne auslasten können, ohne dass die GPU-Auslastung das zeigt.
"""
from __future__ import annotations

import asyncio
import time

_CACHE_TTL = 0.8
_cache: dict | None = None
_cache_at: float = 0.0
_lock = asyncio.Lock()
# Letzter /proc/stat-Sample-Stand je CPU-Zeile ("cpu" = Gesamt, "cpu0".."cpuN"
# = Kerne) - siehe _read_cpu() dazu, warum das modulweit gemerkt werden muss.
_prev_cpu_times: dict[str, tuple[int, int]] = {}


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


def _read_cpu() -> dict:
    """(Gesamt, pro Kern) CPU-Auslastung aus /proc/stat. Die Zeilen dort sind
    NUR kumulative Tick-Zähler seit Systemstart, keine fertigen Prozentwerte
    (anders als nvidia-smi bei der GPU) - eine Auslastung ergibt sich erst
    aus der Differenz zweier Messungen über die Zeit. Um dafür nicht bei
    jedem Aufruf künstlich zu warten (würde das Dashboard-Update verzögern),
    wird hier stattdessen wie bei top/htop der letzte Sample-Stand modulweit
    gemerkt und beim NÄCHSTEN Aufruf die Differenz zum vorherigen gebildet -
    der Abstand zwischen zwei Aufrufen ist ohnehin durch den WebSocket-
    Heartbeat (dashboard.py, ca. 1x/s) vorgegeben, das reicht für eine
    aussagekräftige Momentanauslastung. Der allererste Aufruf nach einem
    Dienststart liefert deshalb None (noch kein vorheriger Stand) - ab dem
    zweiten Aufruf gibt es echte Werte."""
    global _prev_cpu_times
    try:
        with open("/proc/stat") as f:
            lines = [ln for ln in f if ln.startswith("cpu")]
    except OSError:
        return {"cpu_percent": None, "cpu_core_percent": None}

    cur: dict[str, tuple[int, int]] = {}
    for line in lines:
        parts = line.split()
        label = parts[0]
        try:
            nums = [int(p) for p in parts[1:]]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        # Reihenfolge laut /proc/stat: user nice system idle iowait irq
        # softirq steal guest guest_nice - idle+iowait zählt als "nicht
        # arbeitend", der Rest (inkl. steal) als "beschäftigt".
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        cur[label] = (idle, sum(nums))

    def pct(label: str) -> float | None:
        if label not in cur or label not in _prev_cpu_times:
            return None
        idle_now, total_now = cur[label]
        idle_prev, total_prev = _prev_cpu_times[label]
        d_total = total_now - total_prev
        if d_total <= 0:
            return None
        return max(0.0, min(1.0, 1 - (idle_now - idle_prev) / d_total))

    total_pct = pct("cpu")
    core_labels = sorted(
        (l for l in cur if l != "cpu"),
        key=lambda l: int(l[3:]) if l[3:].isdigit() else 0,
    )
    core_pcts = [pct(l) for l in core_labels]
    _prev_cpu_times = cur
    return {"cpu_percent": total_pct, "cpu_core_percent": core_pcts}


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
    mehrfach `nvidia-smi` als Subprozess starten (und nicht mehrfach am
    /proc/stat-Sample für _read_cpu() ziehen - das würde dessen Zeitfenster
    zwischen zwei Messungen künstlich verkürzen)."""
    global _cache, _cache_at
    async with _lock:
        now = time.time()
        if _cache is not None and (now - _cache_at) < _CACHE_TTL:
            return _cache
        gpu = await _read_gpu()
        ram = _read_ram()
        cpu = _read_cpu()
        _cache = {**gpu, **ram, **cpu}
        _cache_at = now
        return _cache
