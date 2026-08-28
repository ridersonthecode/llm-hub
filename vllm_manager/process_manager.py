"""Startet/stoppt vLLM-Engine-Kindprozesse (vllm serve <model>) bei Bedarf -
ohne dass der vllm-Manager-Dienst selbst neu gestartet werden muss.

Hot Pool: bei max_concurrent_models > 1 können mehrere Modelle gleichzeitig
geladen bleiben (je ein Kindprozess auf einem eigenen Port aus dem Pool
engine_port, engine_port+1, ...). Ein Wechsel zwischen bereits geladenen
Modellen ist dann instant statt eines Kaltstarts. Reicht der Speicher (Summe
der gpu_memory_utilization aller Engines, siehe gpu_memory_ceiling) oder die
Poolgröße nicht, wird die am längsten ungenutzte Engine verdrängt (LRU) -
Engines mit gerade aktiven Anfragen werden dabei übersprungen.

Zusätzlich zur reinen Config-Schätzung (Summe der gpu_memory_utilization-Werte
gegen gpu_memory_ceiling) prüft _make_room() vor dem Start noch den TATSÄCHLICH
freien GPU-Speicher (siehe _query_gpu_memory_gib) - die Schätzung sieht nicht,
was Prozesse AUSSERHALB des Hot Pools belegen (z.B. andere GPU-Anwendungen,
oder auf Unified-Memory-Systemen wie NVIDIA GB10 schlicht die allgemeine
System-RAM-Auslastung). Ohne diesen Live-Check startet die Engine trotzdem,
braucht ca. 10s zum Hochfahren und crasht dann mit "Free memory on device X
is less than desired Y" - das war lange Zeit die häufigste Absturzursache.

Verdrängung: reicht Speicherbudget oder Poolgröße nicht, wird die Engine mit
der niedrigsten Priorität (siehe ModelConfig.priority, Gleichstand per LRU)
hart beendet (stop_engine) - ein Neustart braucht dann wieder einen vollen
Kaltstart. (Frühere Versionen boten hier zusätzlich vLLMs Sleep Mode als
schnelleren Mittelweg an - wieder entfernt, siehe Git-Historie: brachte in
der Praxis mehr Ärger als Nutzen, u.a. ein RAM-Vollauf-Absturz, live
beobachtet 2026-08-25.)

Anfrage-Warteschlange (_room_queue_lock): Statt eine Anfrage für ein noch
nicht geladenes Modell sofort mit einem Fehler abzulehnen, wenn gerade kein
Platz ist (alle anderen Engines sind beschäftigt oder selbst im Kaltstart),
wartet sie in einer FIFO-Warteschlange, bis entweder Platz frei wird oder
config.queue_timeout_seconds abläuft. Modellwechsel (unterschiedliche
Zielmodelle) werden dabei strikt nacheinander abgearbeitet - erst wenn eine
Anfrage ihren Platz im Pool gesichert hat, darf die nächste warten anfangen.
Anfragen an ein bereits geladenes ODER gerade ladendes Modell überspringen
diese Warteschlange komplett (siehe ensure_loaded) und laufen wie bisher
nebenläufig."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx

try:
    import psutil
except ImportError:  # pragma: no cover - psutil kommt transitiv über vllm mit
    psutil = None

from . import capability_detector, config_editor, telemetry
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
# Schützt NUR den "dieses Modell braucht eine frisch gestartete Engine"-Pfad
# in ensure_loaded() (siehe Moduldocstring, Anfrage-Warteschlange) - Aufrufer
# für ein bereits geladenes/ladendes/schlafendes Modell fassen diesen Lock
# nie an und werden dadurch nie blockiert. asyncio.Lock() weckt wartende
# Tasks in Ankunftsreihenfolge (FIFO), Modellwechsel werden also automatisch
# strikt nacheinander abgearbeitet.
_room_queue_lock = asyncio.Lock()


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


def _parse_cudagraph_sizes(raw: str) -> list[int]:
    """Parst ModelConfig.cudagraph_capture_sizes ("1, 2,4 8,16" o.ä.) in eine
    aufsteigend sortierte, deduplizierte Liste positiver Ints. Ungültige/leere
    Teile werden stillschweigend übersprungen statt die Engine am Start zu
    hindern - eine verkorkste Eingabe soll bestenfalls zum vLLM-Standard
    zurückfallen, nicht den Modellstart blockieren."""
    sizes: set[int] = set()
    for part in raw.replace(",", " ").split():
        try:
            n = int(part)
        except ValueError:
            continue
        if n > 0:
            sizes.add(n)
    return sorted(sizes)


def _build_command(cfg: Config, model: str, port: int) -> list[str]:
    """Baut das vllm-serve-Kommando. task/vision/tool_calling/reasoning_parser
    kommen NICHT mehr aus mcfg (ModelConfig hat diese Felder seit 2026-08-25
    nicht mehr) - das sind Fakten des Modells selbst, keine Nutzer-Einstellung
    (siehe Nutzer-Feedback: "es ergibt keinen Sinn, Werte anzupassen, die
    feststehen"). Stattdessen wird bei JEDEM Start frisch per
    capability_detector erkannt (dieselbe Heuristik wie der frühere "Auto-
    detect"-Button im Config-Editor, nur jetzt auf jeden Start statt nur auf
    Knopfdruck angewandt) - eine manuelle Änderung an diesen Werten in
    config.json (z.B. von Hand editiert) hätte also ohnehin keine Wirkung
    mehr auf den tatsächlichen Start. Das Ergebnis wird zusätzlich (fire-and-
    forget, blockiert den Start nicht) für die Dashboard-Anzeige/RAG-Task-
    Filter in config.json gespiegelt, siehe config_editor.
    sync_detected_capabilities().

    max_model_len bleibt dagegen ein echter Nutzer-Wert (kleinerer Kontext =
    weniger KV-Cache-Speicherbedarf, eine legitime Abwägung) - wird hier aber
    hart auf die architektonische Obergrenze des Modells (max_position_
    embeddings) gedeckelt, falls die config.json einen höheren Wert enthält
    (z.B. weil das Modell zwischenzeitlich gewechselt oder von Hand
    überschrieben wurde) - höher geht schlicht nicht, unabhängig von der
    Konfiguration."""
    mcfg = cfg.models.get(model)
    gmu, mml = cfg.serve_args_for(model)

    try:
        caps = capability_detector.detect_capabilities(model, cfg.hf_home)
    except Exception:
        logger.exception("Capability-Erkennung für '%s' fehlgeschlagen - starte ohne automatische Flags", model)
        caps = {"found": False}

    if caps.get("found"):
        ceiling = caps["max_model_len"]["suggested"]
        if ceiling and mml > ceiling:
            logger.warning(
                "max_model_len=%d für '%s' liegt über der architektonischen Obergrenze des Modells "
                "(max_position_embeddings=%d) - für diesen Start auf %d gedeckelt.",
                mml, model, ceiling, ceiling,
            )
            mml = ceiling
        # Fire-and-forget: spiegelt die frisch erkannten Werte in config.json,
        # rein für Anzeige/Filter (z.B. RAG-Embedding-Modell-Auswahl nach
        # task=="embed") - beeinflusst NICHT mehr den gerade laufenden Start,
        # der nutzt ausschließlich `caps` direkt (siehe unten).
        asyncio.create_task(config_editor.sync_detected_capabilities(model, caps))

    cmd = [
        cfg.resolved_vllm_bin(), "serve", model,
        "--host", cfg.engine_host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gmu),
        "--max-model-len", str(mml),
    ]
    if caps.get("found"):
        if caps["task"]["suggested"] == "embed":
            # vLLM >=0.26 hat das alte "--task" durch "--runner" ersetzt.
            cmd += ["--runner", "pooling"]
        if caps["tool_calling"]["detected"] and caps["tool_calling"]["suggested_parser"]:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", caps["tool_calling"]["suggested_parser"]]
        if caps["reasoning"]["detected"] and caps["reasoning"]["suggested_parser"]:
            cmd += ["--reasoning-parser", caps["reasoning"]["suggested_parser"]]
    if mcfg:
        if mcfg.fast_load:
            # Siehe ModelConfig.fast_load - deaktiviert CUDA-Graph-Capture,
            # verkürzt den Kaltstart auf Kosten der Dauer-Inferenzgeschwindigkeit.
            cmd.append("--enforce-eager")
        elif mcfg.cudagraph_capture_sizes:
            # Siehe ModelConfig.cudagraph_capture_sizes - schmalere Liste statt
            # vLLMs ~51 Standardgrößen, verkürzt Capture-Zeit ohne Durchsatz im
            # abgedeckten Bereich zu verlieren. Ignoriert bei fast_load (siehe oben).
            sizes = _parse_cudagraph_sizes(mcfg.cudagraph_capture_sizes)
            if sizes:
                cmd += ["--cudagraph-capture-sizes"] + [str(s) for s in sizes]
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


_gpu_mem_query_broken_until = 0.0  # monotonic-Zeitstempel, siehe _query_gpu_memory_gib
# Nach einem Fehlschlag NICHT dauerhaft aufgeben (siehe _query_gpu_memory_gib-
# Docstring: "WICHTIG" 2026-08-27), nur für diese Dauer pausieren und dann
# automatisch erneut versuchen.
_GPU_MEM_QUERY_RETRY_COOLDOWN = 60.0


def _read_mem_available_gib() -> Optional[float]:
    """MemAvailable aus /proc/meminfo in GiB - berücksichtigt (anders als
    MemFree) reclaimbaren Page-Cache/Buffer als tatsächlich verfügbar. Siehe
    _query_gpu_memory_gib() für den Grund, warum das hier gebraucht wird."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1_048_576  # kB -> GiB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _query_gpu_memory_gib() -> tuple[Optional[float], Optional[float]]:
    """(frei, gesamt) in GiB. Basis ist torch.cuda.mem_get_info() - dieselbe
    CUDA-Treiber-API, die vLLM selbst beim Start prüft (siehe die
    "Free memory on device"-Fehlermeldung in worker/utils.py). BEWUSST NICHT
    nvidia-smi/NVML: auf Unified-Memory-Systemen wie NVIDIA GB10 liefert
    `nvidia-smi --query-gpu=memory.free` dort nur "N/A" (per `nvidia-smi -q -d
    MEMORY` verifiziert), torch.cuda.mem_get_info() funktioniert dagegen auch
    dort zuverlässig. Der erste Aufruf initialisiert einen CUDA-Kontext im
    Manager-Prozess (~300-500MB, einmalig) - danach ist die Abfrage praktisch
    kostenlos (<1ms), ein Subprozess pro Aufruf wäre hier unnötiger Overhead.

    WICHTIG (live beobachtet, 2026-08-24 - Modelle gingen bei jedem Wechsel
    unnötig in den Kaltstart): torch.cuda.mem_get_info()
    liefert auf diesem Unified-Memory-System (GB10) einen "frei"-Wert, der
    praktisch exakt /proc/meminfo's MemFree entspricht - NICHT MemAvailable.
    Der Unterschied war live ~47 GiB reclaimbarer Page-Cache, den die CUDA-
    Abfrage fälschlich als "belegt" zählte (genau dasselbe MemFree-vs-
    MemAvailable-Problem, das system_metrics._read_ram() fürs RAM-Dashboard
    schon lösen musste - nur hier in der Verdrängungslogik). Die Folge:
    _make_room() hielt selbst bei reichlich echtem Spielraum ständig zu wenig
    frei für nötig und eskalierte bis zum harten Beenden schlafender Engines,
    obwohl das gar nicht nötig gewesen wäre. Fix: das Maximum aus dem reinen
    CUDA-Wert und MemAvailable verwenden - nie SCHLECHTER als die reine
    CUDA-Zahl (falls /proc/meminfo mal nicht lesbar ist oder dieses konkrete
    Deployment doch auf einer klassischen dGPU statt Unified Memory läuft),
    aber korrigiert das Unterschätzen auf genau diesem Hardware-Typ.

    WICHTIG (live beobachtet, 2026-08-27 - Chat "Kommunikation mit LLM-
    Modellen prüfen"): torch.cuda.mem_get_info() aus einem zweiten Prozess
    heraus, während eine Engine bereits fast den gesamten Unified-Memory-Pool
    belegt (genau die Situation, in der DIESER Live-Check am wichtigsten
    wäre!), kann selbst mit einem harmlosen "CUDA error: out of memory"
    fehlschlagen (der kleine CUDA-Kontext, den der Aufruf einmalig braucht,
    passt dann nicht mehr) - reproduziert live durch zwei Aufrufe kurz
    hintereinander, einer schlug fehl, der andere (bei etwas mehr Luft)
    lief durch. Ein einzelner solcher Ausrutscher darf den Live-Check daher
    NICHT für den Rest der Prozesslaufzeit abschalten (das war der alte
    Bug: `_gpu_mem_query_broken = True` blieb permanent gesetzt) - sonst
    startet jedes folgende Modell blind gegen die konfigurierten
    gpu_memory_utilization-Werte, crasht bei einem zu knappen Wert erst
    nach vollem Kaltstart mit dem KV-Cache-Defizit-Fehler und braucht dann
    mehrere komplette Neuversuche (bei großen Checkpoints jeweils mehrere
    Minuten) statt vorher schon zu wissen, dass es eng wird. Fix: nur
    kurz pausieren (_GPU_MEM_QUERY_RETRY_COOLDOWN), dann automatisch
    erneut versuchen, statt für immer aufzugeben."""
    global _gpu_mem_query_broken_until
    now = time.monotonic()
    if now < _gpu_mem_query_broken_until:
        return None, None
    try:
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        mem_available_gib = _read_mem_available_gib()
        if mem_available_gib is not None:
            free_gib = min(max(free_gib, mem_available_gib), total_gib)
        return free_gib, total_gib
    except Exception:
        logger.warning(
            "Live-GPU-Speichercheck fehlgeschlagen (torch.cuda.mem_get_info) - "
            "verlasse mich für die nächsten %d Sekunden nur auf die konfigurierten "
            "gpu_memory_utilization-Werte (gpu_memory_ceiling), danach automatisch "
            "neuer Versuch (siehe Docstring - kein Dauerzustand mehr).",
            int(_GPU_MEM_QUERY_RETRY_COOLDOWN),
            exc_info=True,
        )
        _gpu_mem_query_broken_until = now + _GPU_MEM_QUERY_RETRY_COOLDOWN
        return None, None


async def _make_room(cfg: Config, model: str, gmu: float) -> None:
    """Verdrängt bei Bedarf Engines (LRU), damit `model` (mit geschätztem
    Speicherbedarf `gmu`) ins Poolgrößen- und Speicherbudget passt. Engines mit
    gerade aktiven Anfragen werden NIE automatisch verdrängt - das würde eine
    laufende Antwort mitten drin abbrechen. Ebenfalls geschützt: Engines, die
    gerade selbst noch im Kaltstart stecken (state == "loading") - live
    beobachtet (2026-08-24): ohne diesen Schutz kann ein GLEICHZEITIGER
    Request für ein ANDERES Modell so eine frisch gestartete, noch ladende
    Engine als LRU-Opfer verdrängen, bevor sie überhaupt fertig ist. Die
    Engine reagiert während der frühen CUDA-Initialisierung oft nicht
    rechtzeitig auf SIGTERM, wird nach 30s per SIGKILL beendet, und der
    URSPRÜNGLICHE Aufrufer bekommt einen irreführenden "vLLM-Engine ist
    unerwartet beendet (exit=-9)"-Fehler, obwohl am Modell selbst nichts
    kaputt war.

    Statt sofort mit einem Fehler aufzugeben, wenn gerade nichts Verdrängbares
    da ist (alle übrigen Engines beschäftigt/ladend), WARTET diese Funktion
    bis zu config.queue_timeout_seconds (siehe Moduldocstring, Anfrage-
    Warteschlange) und versucht es währenddessen alle 1s erneut - nur wenn
    das Modell selbst mit leerem Pool nicht passt (Budget-Problem, keine
    Frage der Wartezeit), oder der Timeout ohne Erfolg abläuft, wird
    tatsächlich ein RuntimeError geworfen.

    Verdrängung bedeutet hier immer: Engine komplett beenden (stop_engine) -
    ein späterer erneuter Bedarf braucht dann wieder einen vollen Kaltstart.
    (Frühere Versionen boten hier zusätzlich vLLMs Sleep Mode als schnelleren
    Mittelweg an - wieder entfernt, siehe Git-Historie: brachte in der Praxis
    mehr Ärger als Nutzen, u.a. einen RAM-Vollauf-Absturz, live beobachtet
    2026-08-25.)"""
    deadline = time.time() + cfg.queue_timeout_seconds

    def others() -> list[EngineState]:
        return [e for e in engines.values() if e.model != model]

    def protected_models() -> set[str]:
        return (
            {r["model"] for r in telemetry.active_requests.values()}
            | {e.model for e in engines.values() if e.state == "loading" and e.model != model}
        )

    def total_util() -> float:
        return sum(cfg.serve_args_for(e.model)[0] for e in others()) + gmu

    async def evict_one() -> bool:
        """Verdrängt EINE Engine per Priorität+LRU. True bei Erfolg, False
        wenn gerade nichts Verdrängbares da ist (alle übrigen Engines sind
        geschützt)."""
        candidates = [e for e in others() if e.model not in protected_models()]
        if not candidates:
            return False
        # Niedrigste Priorität zuerst (siehe ModelConfig.priority - höhere
        # Zahl = wird später verdrängt), bei Gleichstand wie bisher LRU
        # (am längsten ungenutzt zuerst).
        victim = min(candidates, key=lambda e: (cfg.priority_for(e.model), e.last_used))
        logger.info("Verdränge Modell '%s' aus dem Pool, um Platz für '%s' zu schaffen", victim.model, model)
        await stop_engine(victim.model, reason=f"evicted_for:{model}")
        return True

    async def wait_for_room(busy_msg: str, impossible_msg: str) -> None:
        """Wie evict_one(), aber wartet bei Bedarf in der Warteschlange (siehe
        Moduldocstring) statt sofort aufzugeben."""
        while True:
            pool = others()
            if not pool:
                raise RuntimeError(impossible_msg)
            if await evict_one():
                return
            if time.time() > deadline:
                raise RuntimeError(
                    f"Kein Platz im Hot Pool für '{model}' nach {cfg.queue_timeout_seconds}s Warten in "
                    f"der Warteschlange: {busy_msg} Bitte manuell ein Modell entladen (Dashboard-Button "
                    f"oder POST /models/<model>/unload) und die Anfrage erneut senden."
                )
            await asyncio.sleep(1.0)

    # Phase 1: Poolgröße (max_concurrent_models) und Config-Schätzung
    # (gpu_memory_ceiling) - billig, kein GPU-Zugriff pro Anfrage nötig.
    while len(others()) >= cfg.max_concurrent_models or total_util() > cfg.gpu_memory_ceiling:
        busy = sorted({e.model for e in others() if e.model in protected_models()})
        busy_msg = (
            f"alle {len(others())} aktuell geladenen Modelle ({', '.join(busy)}) haben durchgehend "
            f"eine laufende Anfrage oder stecken selbst im Kaltstart."
        )
        impossible_msg = (
            f"Modell '{model}' passt nicht ins GPU-Speicherbudget "
            f"(gpu_memory_utilization={gmu} vs. gpu_memory_ceiling={cfg.gpu_memory_ceiling}), "
            f"auch mit leerem Hot Pool nicht. gpu_memory_ceiling oder die "
            f"gpu_memory_utilization dieses Modells in config.json anpassen."
        )
        await wait_for_room(busy_msg, impossible_msg)

    # Live-Check: siehe Moduldocstring oben / _query_gpu_memory_gib(). Läuft
    # ERST NACH der schnellen Config-Schätzung (kein Overhead im Normalfall),
    # verdrängt bei Bedarf aber weitere Engines - auch wenn die Schätzung oben
    # bereits "passt" sagte, weil z.B. ein fremder Prozess (ComfyUI o.ä.) oder
    # die allgemeine Systemauslastung (Unified Memory) Speicher belegt, den
    # gpu_memory_ceiling nicht kennt.
    while True:
        free_gib, total_gib = await asyncio.to_thread(_query_gpu_memory_gib)
        if free_gib is None:
            return  # torch/CUDA nicht verfügbar - wie bisher nur auf die Config-Schätzung verlassen
        needed_gib = gmu * total_gib
        if free_gib >= needed_gib:
            return
        pool = others()
        logger.info(
            "Live-Speicherprüfung für '%s': nur %.1f/%.1f GiB frei (Defizit %.1f GiB) - Pool: %s",
            model, free_gib, needed_gib, needed_gib - free_gib,
            [(e.model, e.state, round(cfg.serve_args_for(e.model)[0], 2)) for e in pool],
        )
        busy_msg = (
            f"nur {free_gib:.1f}/{needed_gib:.1f} GiB GPU-Speicher frei, alle {len(pool)} übrigen "
            f"geladenen Modelle ({', '.join(sorted({e.model for e in pool}))}) haben durchgehend "
            f"eine laufende Anfrage oder stecken selbst im Kaltstart. Das Defizit stammt vermutlich "
            f"von Speicher, den ANDERE Prozesse belegen (nicht vom Hot Pool verwaltet)."
        )
        impossible_msg = (
            f"Modell '{model}' passt aktuell nicht in den GPU-Speicher: nur {free_gib:.1f} GiB frei, "
            f"benötigt werden ca. {needed_gib:.1f} GiB (gpu_memory_utilization={gmu}). Der Hot Pool ist "
            f"bereits leer - das Defizit stammt vermutlich von Speicher, den ANDERE Prozesse belegen "
            f"(nicht vom Hot Pool verwaltet, z.B. ComfyUI/Remote-Desktop, oder allgemeine Systemauslastung "
            f"auf diesem Unified-Memory-System) - bitte manuell Speicher freigeben oder "
            f"gpu_memory_utilization/gpu_memory_ceiling in config.json senken."
        )
        await wait_for_room(busy_msg, impossible_msg)
        # CUDA-Treiber braucht nach Prozessende einen kurzen Moment, um den
        # freigegebenen Speicher tatsächlich als frei zu melden.
        await asyncio.sleep(1.0)


def _collect_children(pid: int) -> list:
    """Best-effort Snapshot der (rekursiven) Kindprozesse EINES Engine-Prozesses,
    VOR dessen Beendigung - psutil.Process.children() liefert nichts mehr,
    sobald der Elternprozess schon weg ist, deshalb muss das vorher passieren.
    vLLM startet je nach Konfiguration eigene Multiprocessing-Worker
    (EngineCore u.ä.) als Kindprozesse von "vllm serve"; die sterben nicht
    immer zuverlässig mit dem Eltern-SIGTERM mit."""
    if psutil is None:
        return []
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


async def _reap_leftover_children(children: list, timeout: float = 5.0) -> None:
    """Beendet Kindprozesse, die nach dem Stoppen der eigentlichen Engine noch
    übrig sind (siehe _collect_children) - das ist die Hauptursache für
    "unsauberes Unloading": der getrackte "vllm serve"-Prozess ist weg, aber
    seine eigenen Worker-Subprozesse laufen als Waisen weiter, belegen GPU-
    Speicher und tauchen im Dashboard nicht mehr auf (kein EngineState mehr
    dafür)."""
    if psutil is None or not children:
        return
    alive = [c for c in children if c.is_running()]
    if not alive:
        return
    logger.warning(
        "Räume %d übrig gebliebene(n) Kind-Prozess(e) nach Engine-Stop auf (pids=%s) - "
        "vLLM hat sie beim eigenen Beenden nicht mitgenommen",
        len(alive), [c.pid for c in alive],
    )

    def _do() -> None:
        for c in alive:
            try:
                c.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, still = psutil.wait_procs(alive, timeout=timeout)
        for c in still:
            try:
                c.kill()
            except psutil.NoSuchProcess:
                pass

    await asyncio.to_thread(_do)


def _matched_vllm_serve_model(cmdline: list[str], vllm_bin: Optional[str]) -> Optional[str]:
    """Gibt den Modellnamen zurück, wenn `cmdline` ein "vllm serve <model> ..."-
    Aufruf DIESER Installation ist (schützt davor, versehentlich einen fremden
    vLLM-Prozess - z.B. eine andere Installation/venv wie .venv-quantize - zu
    beenden), sonst None.

    _build_command() übergibt zwar [vllm_bin, "serve", model, ...] an
    create_subprocess_exec, aber die "vllm"-Datei hat selbst ein
    "#!/.../python3"-Shebang - der Kernel schreibt argv beim exec deshalb um.
    Tatsächlich in ps/psutil sichtbar (live verifiziert):
    ['.../python3', '.../vllm', 'serve', <model>, ...], NICHT wie übergeben.
    Daher wird hier an Position 0 UND 1 nach dem vllm-Binary gesucht statt nur
    an Position 0."""
    for i, arg in enumerate(cmdline[:2]):
        if os.path.basename(arg) != "vllm":
            continue
        if len(cmdline) <= i + 2 or cmdline[i + 1] != "serve":
            continue
        if vllm_bin is not None:
            try:
                if os.path.realpath(arg) != vllm_bin:
                    continue
            except OSError:
                continue
        return cmdline[i + 2]
    return None


async def reap_orphan_engines() -> list[dict]:
    """Findet und beendet vLLM-Engine-Prozesse ("vllm serve ..." dieser
    Installation), die NICHT im aktuellen `engines`-Dict getrackt sind, samt
    ihrer Kindprozesse. Das sind Reste eines Absturzes des Manager-Prozesses
    selbst (der `engines`-Zustand lebt nur im Speicher und ist nach einem
    Neustart leer, die zuvor gestarteten Kindprozesse überleben aber) oder
    eines unsauberen Beendens - genau die Prozesse, die "im Dashboard nicht
    auftauchen" (kein EngineState mehr da), aber weiter GPU-Speicher/Port
    belegen. Wird beim Start (lifespan) und danach periodisch aufgerufen
    (siehe main.py)."""
    if psutil is None:
        return []
    cfg = get_config()
    try:
        vllm_bin = os.path.realpath(cfg.resolved_vllm_bin())
    except OSError:
        vllm_bin = None
    tracked_pids = {e.process.pid for e in engines.values() if e.process is not None}
    own_pid = os.getpid()

    def _scan() -> list[tuple]:
        found = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid in tracked_pids or pid == own_pid:
                    continue
                cmdline = proc.info["cmdline"] or []
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            model = _matched_vllm_serve_model(cmdline, vllm_bin)
            if model is not None:
                found.append((proc, model))
        return found

    orphans = await asyncio.to_thread(_scan)
    killed = []
    for proc, model in orphans:
        try:
            pid = proc.pid
            created_at = proc.create_time()
        except psutil.NoSuchProcess:
            continue
        logger.warning(
            "Verwaister vLLM-Engine-Prozess gefunden (pid=%s, Modell=%s, seit %.0fs), "
            "der zu keinem aktuellen EngineState gehört - vermutlich Rest eines Manager-"
            "Absturzes oder unsauberen Beendens. Wird beendet.",
            pid, model, time.time() - created_at,
        )
        children = _collect_children(pid)
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

        def _wait_and_kill(root=proc, kids=children) -> None:
            _gone, alive = psutil.wait_procs([root, *kids], timeout=10.0)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

        await asyncio.to_thread(_wait_and_kill)
        model_history.appendleft({
            "model": model,
            "loaded_at": created_at,
            "unloaded_at": time.time(),
            "duration_seconds": round(time.time() - created_at, 1),
            "reason": "orphan_reaped",
            "error": f"Verwaister Prozess (pid={pid}) ohne Tracking im aktuellen Manager gefunden und beendet.",
        })
        killed.append({"pid": pid, "model": model})
    return killed


async def stop_engine(model: Optional[str] = None, reason: str = "manual_unload", timeout: float = 30.0) -> None:
    """Beendet eine laufende Engine und schreibt einen Verlaufseintrag.
    Ohne `model` werden ALLE laufenden Engines beendet (z.B. beim Dienst-
    Shutdown). `reason` z.B. "manual_unload", "idle_timeout", "crashed",
    "timeout", "shutdown", "orphan_reaped" oder "evicted_for:<neues_modell>"."""
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
        # VOR dem Beenden einsammeln (siehe _collect_children) - danach ist der
        # Elternprozess weg und psutil kann seine Kinder nicht mehr auflisten.
        children = _collect_children(proc.pid)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Engine reagiert nicht auf SIGTERM, sende SIGKILL")
            proc.kill()
            await proc.wait()
        # Fängt vLLM-Versionen ab, die ihre eigenen Worker-Subprozesse beim
        # Beenden nicht zuverlässig mitnehmen (siehe _reap_leftover_children) -
        # sonst genau die Waisenprozesse, die "unsauberes Unloading" verursachen.
        await _reap_leftover_children(children)
    engines.pop(model, None)


def is_model_enabled(cfg: Config, model: str) -> tuple[bool, Optional[str]]:
    mcfg = cfg.models.get(model)
    if mcfg is not None and not mcfg.enabled:
        return False, mcfg.notes or f"Modell '{model}' ist in config.json als enabled:false markiert."
    return True, None


_KV_CACHE_DEFICIT_RE = re.compile(
    r"([\d.]+)\s*GiB KV cache is needed, which is larger than the available "
    r"KV cache memory \(([\d.]+)\s*GiB\)"
)
# Fallback-Muster: manchmal liefert vLLM nur die generische Meldung "No
# available memory for the cache blocks" OHNE die obigen Zahlen (live
# beobachtet, 2026-08-24) - der eigentliche Fehlbetrag steht dann einige
# Zeilen VORHER als negativer Zwischenwert ("Available KV cache memory:
# -8.67 GiB"), den vLLM beim Speicher-Profiling loggt, bevor es überhaupt
# zum eigentlichen Fehler kommt.
_KV_CACHE_NEGATIVE_RE = re.compile(r"Available KV cache memory:\s*(-[\d.]+)\s*GiB")


def _parse_kv_cache_deficit_gib(text: str) -> Optional[float]:
    """Extrahiert aus vLLMs eigenen Logzeilen beim Start das KV-Cache-Defizit
    in GiB - None, falls keines der beiden bekannten Muster gefunden wird
    (andere Fehlerursache, siehe Aufrufer). Zwei Formen, siehe Kommentare an
    den Regexes oben: die präzise "X GiB needed vs. Y GiB available"-Meldung,
    oder ersatzweise der negative Zwischenwert aus dem Speicher-Profiling."""
    m = _KV_CACHE_DEFICIT_RE.search(text)
    if m:
        needed, available = float(m.group(1)), float(m.group(2))
        deficit = needed - available
        if deficit > 0:
            return deficit
    m2 = _KV_CACHE_NEGATIVE_RE.search(text)
    if m2:
        return abs(float(m2.group(1)))
    return None


async def _autocorrect_kv_cache_deficit(model: str, error_message: str) -> bool:
    """Statt gpu_memory_utilization im Voraus zu erraten (fehleranfällig -
    hängt von Layer-Anzahl, KV-Head-Zahl, dtype, Hybrid-Architekturen wie
    Mamba/GatedDeltaNet-Mischungen ab, siehe capability_detector), lässt sich
    das erste Fehlschlagen einfach vLLM selbst überlassen: dessen eigene
    Fehlermeldung beim Start nennt den exakt fehlenden KV-Cache-Speicher in
    GiB (siehe _parse_kv_cache_deficit_gib) - autoritativer als jede eigene
    Schätzung, weil vLLM zu diesem Zeitpunkt bereits die echte Modell-
    Architektur geladen hat.

    Korrigiert IMMER, wenn dieses eindeutige Fehlermuster erkannt wird - auch
    wenn der aktuelle Wert vom Nutzer manuell gesetzt war (anders als z.B.
    register_model_if_missing, das nie einen bestehenden Wert anfasst): dort
    ist es reine Vorsorge ohne jeden Beweis, hier liegt ein ECHTER,
    reproduzierbarer Absturz vor - ein manuell gesetzter, nachweislich zu
    knapper Wert lässt das Modell sonst dauerhaft unladbar bleiben, bis
    jemand von Hand nachbessert. Schreibt den korrigierten Wert (Defizit +
    15% Sicherheitsaufschlag, gedeckelt auf 0.95) direkt in config.json.
    Gibt True zurück, wenn korrigiert wurde (der Aufrufer soll dann einen
    erneuten Versuch starten)."""
    cfg = get_config()
    deficit_gib = _parse_kv_cache_deficit_gib(error_message)
    if deficit_gib is None:
        # error_message enthält nur den kurzen 40-Zeilen-Tail (siehe _tail()
        # in _ensure_loaded_once) - der negative Zwischenwert aus dem
        # Speicher-Profiling kann weiter zurückliegen und dort schon
        # rausgefallen sein. Bei Bedarf gezielt einen größeren Ausschnitt
        # direkt aus der Logdatei nachlesen (derselbe Pfad wie beim Start).
        log_path = LOG_DIR / f"{_safe_name(model)}.log"
        wider_tail = await asyncio.to_thread(_tail, log_path, 200)
        deficit_gib = _parse_kv_cache_deficit_gib(wider_tail)
    if deficit_gib is None:
        return False

    _free_gib, total_gib = await asyncio.to_thread(_query_gpu_memory_gib)
    if not total_gib:
        return False

    current_gmu, _ = cfg.serve_args_for(model)
    # +35% statt nur des reinen Defizits: vLLMs CUDA-Graph-Speicherprofiling
    # frisst einen Teil jeder gpu_memory_utilization-Erhöhung selbst wieder
    # auf (live beobachtet - eine Erhöhung um genau das gemeldete Defizit
    # ließ beim erneuten Versuch fast dasselbe Restdefizit übrig), ohne
    # großzügigeren Aufschlag bräuchte es öfter mehrere Korrekturrunden
    # (siehe MAX_KV_CACHE_AUTOCORRECT_ATTEMPTS in ensure_loaded - die fängt
    # das zwar ab, aber jeder Fehlversuch kostet bei großen Modellen
    # mehrere Minuten Kaltstart-Zeit für nichts).
    new_gmu = min(0.95, round(current_gmu + (deficit_gib * 1.35) / total_gib, 3))
    if new_gmu <= current_gmu:
        return False

    logger.warning(
        "Modell '%s': gpu_memory_utilization=%.3f reichte nicht für den konfigurierten "
        "max_model_len (KV-Cache-Defizit laut vLLM: %.2f GiB) - korrigiere automatisch "
        "auf %.3f, trage das in config.json ein und versuche erneut zu laden.",
        model, current_gmu, deficit_gib, new_gmu,
    )

    def _mutate(dump: dict):
        # Der obige cfg-Snapshot kann durch die Zeit für Log-Tail-Read und
        # GPU-Abfrage schon wieder veraltet sein (siehe config_editor.
        # patch_config()-Docstring - Auslöser dieses Mechanismus war genau
        # diese Funktion). Deshalb hier NICHT den oben berechneten new_gmu
        # blind draufschreiben, sondern gegen den gerade frisch von der
        # Platte gelesenen Wert neu rechnen - falls der z.B. durch einen
        # parallelen manuellen Fix im Dashboard-Editor inzwischen schon
        # ausreicht, gibt es nichts mehr zu tun.
        fresh_gmu, _ = Config(**dump).serve_args_for(model)
        recomputed_new_gmu = min(0.95, round(fresh_gmu + (deficit_gib * 1.35) / total_gib, 3))
        if recomputed_new_gmu <= fresh_gmu:
            return False
        entry = dump["models"].setdefault(model, {})
        entry["gpu_memory_utilization"] = recomputed_new_gmu
        old_notes = (entry.get("notes") or "").strip()
        correction_note = (
            f"gpu_memory_utilization automatisch von {fresh_gmu} auf {recomputed_new_gmu} korrigiert "
            f"(KV-Cache-Defizit {deficit_gib:.2f} GiB laut vLLM-Fehlermeldung beim Start)."
        )
        entry["notes"] = f"{old_notes}\n{correction_note}".strip() if old_notes else correction_note

    try:
        result = await asyncio.to_thread(config_editor.patch_config, _mutate)
    except Exception:
        logger.exception("Konnte automatisch korrigiertes gpu_memory_utilization für '%s' nicht speichern", model)
        return False
    return result is not None


MAX_KV_CACHE_AUTOCORRECT_ATTEMPTS = 3


async def ensure_loaded(model: str, wait: bool = True) -> dict:
    """Dünner Wrapper um _ensure_loaded_once() mit automatischer Selbst-
    korrektur: schlägt der Kaltstart speziell an einem zu knappen KV-Cache-
    Speicherbudget fehl (siehe _autocorrect_kv_cache_deficit), wird
    gpu_memory_utilization korrigiert und erneut versucht - bis zu
    MAX_KV_CACHE_AUTOCORRECT_ATTEMPTS mal, NICHT nur einmal: live beobachtet
    (2026-08-24), dass EINE Korrektur manchmal nicht reicht, weil vLLMs
    CUDA-Graph-Speicherprofiling einen Teil jeder Erhöhung selbst mit
    aufbraucht (nichtlinearer Effekt, siehe dessen eigener Hinweis "CUDA
    graph memory profiling..." im Log) - das Restdefizit wird dadurch mit
    jedem Versuch kleiner, aber nicht notwendigerweise beim ersten Mal schon
    Null. _autocorrect_kv_cache_deficit() bricht selbst ab (gibt False
    zurück), sobald keine Verbesserung mehr möglich ist (0.95-Deckel erreicht
    oder kein positives Defizit mehr erkennbar), das begrenzt die Schleife
    zusätzlich zum harten Attempt-Limit hier. Jeder andere Fehler (Modell
    nicht gefunden, Timeout, ...) propagiert unverändert nach dem ersten
    Versuch."""
    for attempt in range(1, MAX_KV_CACHE_AUTOCORRECT_ATTEMPTS + 1):
        try:
            return await _ensure_loaded_once(model, wait)
        except RuntimeError as e:
            if attempt >= MAX_KV_CACHE_AUTOCORRECT_ATTEMPTS or not await _autocorrect_kv_cache_deficit(model, str(e)):
                raise
    raise AssertionError("unreachable")  # for-Schleife kehrt immer per return/raise zurück


async def _ensure_loaded_once(model: str, wait: bool = True) -> dict:
    """Lädt `model`, falls nötig, und wartet (falls `wait`) bis es bereit ist.

    WICHTIG (war lange ein Bug): der Pool-Lock (`_pool_lock`) darf NUR die
    kurzen, zustandsverändernden Schritte schützen (Prüfen/Verdrängen/
    Prozess-Start) - NICHT die eigentliche Warteschleife auf "gesund", die bei
    großen Modellen mehrere Minuten bis zu `startup_timeout_seconds` (aktuell
    1800s) dauern kann. Vorher lag die komplette Warteschleife INNERHALB von
    `async with _pool_lock`, wodurch JEDER andere ensure_loaded()-Aufruf -
    selbst für ein bereits fertig geladenes, anderes Modell (z.B. das
    RAG-Embedding-Modell beim Text hinzufügen, siehe rag.py) - für die gesamte
    Dauer eines FREMDEN Kaltstarts blockiert hat. Das hat den Hot Pool
    (mehrere Modelle gleichzeitig nutzbar) faktisch ausgehebelt, sobald
    irgendein Modell gerade kalt startet. Fix: Lock wird direkt nach dem
    Prozess-Start wieder freigegeben, die Warteschleife läuft außerhalb davon -
    mehrere gleichzeitige Aufrufer für DASSELBE noch ladende Modell warten
    einfach auf denselben EngineState statt es doppelt zu starten.

    Anfrage-Warteschlange (siehe Moduldocstring): Nur der Zweig, der eine
    WIRKLICH NEUE Engine braucht, fasst `_room_queue_lock` an - Aufrufer für
    ein bereits geladenes oder ladendes Modell überspringen ihn komplett und
    werden von einem parallel wartenden Modellwechsel nicht ausgebremst."""
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
        if existing is not None and existing.state == "loading":
            # Ein anderer Aufruf lädt dieses Modell bereits (z.B. zwei fast
            # gleichzeitige Chat-Requests) - nicht doppelt anfassen, sondern
            # weiter unten (außerhalb des Locks) denselben Kaltstart abwarten.
            eng = existing
        else:
            eng = None

    if eng is None:
        # Braucht eine frisch gestartete Engine - dafür FIFO-Warteschlange
        # (siehe Moduldocstring): Modellwechsel werden strikt nacheinander
        # abgearbeitet statt sich gegenseitig um Platz zu streiten oder
        # sofort mit einem Fehler abgewiesen zu werden. WICHTIG: _make_room()
        # kann jetzt (Anfrage-Warteschlange) lange warten - das darf NICHT
        # innerhalb von _pool_lock passieren, sonst blockiert das jeden
        # anderen ensure_loaded()-Aufruf (selbst für bereits fertige Modelle)
        # für die gesamte Wartezeit, siehe Docstring-Warnung oben. _pool_lock
        # schützt deshalb hier nur noch die beiden kurzen Rand-Schritte
        # (erneute Prüfung + eigentlicher Prozess-Start), nicht das Warten
        # dazwischen.
        async with _room_queue_lock:
            need_start = False
            async with _pool_lock:
                # Erneut prüfen: während des Wartens auf _room_queue_lock
                # könnte ein anderer Aufruf (z.B. zwei fast gleichzeitige
                # Requests für dasselbe NEUE Modell) das Modell inzwischen
                # schon gestartet haben.
                existing = engines.get(model)
                if existing is not None and is_ready(model):
                    existing.last_used = time.time()
                    _persist_last_active(model)
                    return existing.status()
                if existing is not None and existing.state == "loading":
                    eng = existing
                else:
                    if existing is not None:
                        # Hängender/abgestürzter Eintrag - erst aufräumen, dann sauber neu starten.
                        await stop_engine(model, reason="restart")
                    need_start = True

            if need_start:
                gmu, mml = cfg.serve_args_for(model)
                await _make_room(cfg, model, gmu)  # kann warten - bewusst AUSSERHALB von _pool_lock

                async with _pool_lock:
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
        # _room_queue_lock ab hier freigegeben - der nächste wartende
        # Modellwechsel darf jetzt versuchen, seinen Platz zu sichern.

    if not wait:
        return eng.status()

    deadline = time.time() + cfg.startup_timeout_seconds
    while time.time() < deadline:
        if eng.process.returncode is not None:
            tail = _tail(eng.log_path)
            msg = f"vLLM-Engine für '{model}' ist unerwartet beendet (exit={eng.process.returncode})."
            eng.last_error = msg
            await stop_engine(model, reason="crashed")
            raise RuntimeError(f"{msg}\nLetzte Log-Zeilen ({eng.log_path}):\n{tail}")
        if await _health_ok(cfg, eng.port):
            eng.state = "ready"
            eng.last_used = time.time()
            _persist_last_active(model)
            # Fire-and-forget: ein per direktem Request geladenes, bisher
            # nicht in config.json registriertes Modell (z.B. vLLM hat es
            # selbst automatisch von HF heruntergeladen) wird jetzt
            # nachgetragen - siehe config_editor.register_model_if_missing.
            # Bewusst NICHT awaited, damit das die Antwort an den
            # wartenden Aufrufer nicht verzögert.
            asyncio.create_task(config_editor.register_model_if_missing(
                model,
                note="Automatisch registriert beim ersten erfolgreichen Laden - "
                     "war lokal gecacht oder wurde von vLLM selbst von HF "
                     "heruntergeladen, ohne vorher im Config-Editor angelegt zu sein. "
                     "Bitte Werte prüfen (nur automatisch erkannt) - gpu_memory_utilization "
                     "ist ein konservativer Minimalwert zum sicheren Start, für mehr Kontext/"
                     "Durchsatz ggf. erhöhen.",
            ))
            return eng.status()
        await asyncio.sleep(2)

    tail = _tail(eng.log_path)
    msg = f"Timeout beim Start von '{model}' nach {cfg.startup_timeout_seconds}s."
    eng.last_error = msg
    await stop_engine(model, reason="timeout")
    raise TimeoutError(f"{msg}\nLetzte Log-Zeilen ({eng.log_path}):\n{tail}")
