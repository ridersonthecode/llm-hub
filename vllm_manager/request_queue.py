"""Globale Obergrenze für gleichzeitig laufende Anfragen - über ALLE Modelle
hinweg, unabhängig vom Hot Pool (der begrenzt nur, wie viele Modelle gleichzeitig
GELADEN sind, nicht wie viele Anfragen gleichzeitig BEARBEITET werden). Siehe
config.max_concurrent_requests/queue_debounce_seconds (config.py).

Warum debounced statt sofort bei freiem Slot zu übergeben (siehe Chat vom
2026-08-27): Clients wie VS Code/GitHub Copilot Chat schicken bei einem
Werkzeugaufruf/Retry/Abbruch-und-Neuversuch oft mehrere Anfragen kurz
hintereinander (Millisekunden bis wenige Sekunden Abstand). Würde eine
wartende Anfrage sofort losgeschickt, sobald zufällig ein Slot frei wird,
könnte das genau die Anfrage sein, die der Client Sekundenbruchteile später
selbst schon wieder verwirft (z.B. weil der Nutzer inzwischen weitergetippt
hat) - verschwendete Arbeit (ggf. sogar ein Modell-Kaltstart) für nichts.
Deshalb: eine wartende Anfrage wird erst an ein Modell übergeben, nachdem
GLOBAL (über alle Anfragen hinweg, nicht nur diese eine) für mindestens
queue_debounce_seconds KEINE neue Anfrage mehr eingetroffen ist - ein simples,
robustes Anti-Burst-Debounce statt Client-spezifischer Heuristik (die einen
API-Key oder eine Client-ID bräuchte, die wir nicht zuverlässig haben).

Bewusst als eigenes Modul statt in process_manager.py: diese Warteschlange
kennt keine Modelle, keine Engines, keinen GPU-Speicher - sie zählt nur
"wie viele Anfragen sind gerade in main.py/ollama_compat.py unterwegs" und
gate't den Übergang zu process_manager.ensure_loaded(). Betrifft NUR echte
eingehende Client-Anfragen (proxy_v1/ollama_compat.api_chat) - interne Aufrufe
von ensure_loaded() (RAG-Embedding-Lookups, Performance-Tuner-Benchmarks, MCP-
Tools) laufen bewusst NICHT durch dieses Gate, sonst würden Hintergrund-
Vorgänge echte Nutzeranfragen ausbremsen oder umgekehrt selbst unnötig
warten."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Optional

from . import telemetry
from .config import get_config

logger = logging.getLogger("vllm_manager.request_queue")

# Wie oft der Hintergrund-Dispatcher prüft, ob eine wartende Anfrage jetzt
# übergeben werden darf (Slot frei UND Ruhephase erreicht). Kein Grund, das
# konfigurierbar zu machen - kleiner als jeder sinnvolle queue_debounce_seconds-
# Wert, sorgt also nie selbst für spürbare zusätzliche Verzögerung.
_POLL_INTERVAL = 0.25

_active = 0
_waiters: "OrderedDict[str, asyncio.Future]" = OrderedDict()
_last_arrival = time.monotonic()
_lock: Optional[asyncio.Lock] = None
_dispatcher_task: Optional[asyncio.Task] = None


def _get_lock() -> asyncio.Lock:
    # Lazy statt Modul-Ebene: asyncio.Lock() bindet sich an die Event-Loop zum
    # Erstellungszeitpunkt - bei Import (vor Start der Loop) angelegt, würde er
    # an die falsche Loop gebunden sein (z.B. in Tests mit eigener Loop).
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _ensure_dispatcher() -> None:
    global _dispatcher_task
    if _dispatcher_task is None or _dispatcher_task.done():
        _dispatcher_task = asyncio.create_task(_dispatch_loop())


async def _dispatch_loop() -> None:
    while True:
        await asyncio.sleep(_POLL_INTERVAL)
        try:
            await _try_dispatch()
        except Exception:
            logger.exception("Fehler im Request-Queue-Dispatcher (Schleife läuft weiter)")


async def _try_dispatch() -> None:
    global _active
    async with _get_lock():
        if not _waiters:
            return
        cfg = get_config()
        limit = max(1, cfg.max_concurrent_requests)
        debounce = max(0.0, cfg.queue_debounce_seconds)
        while _waiters and _active < limit:
            if (time.monotonic() - _last_arrival) < debounce:
                # Noch keine Ruhe seit der letzten (irgendeiner!) eingegangenen
                # Anfrage - erst wieder beim nächsten Poll-Tick prüfen.
                break
            rid, fut = next(iter(_waiters.items()))
            del _waiters[rid]
            _active += 1
            if not fut.done():
                fut.set_result(None)


async def acquire(rid: str) -> None:
    """Blockiert, bis für `rid` ein Slot frei ist (siehe Moduldocstring für das
    Debounce-Verhalten). Muss von genau einem passenden release() gefolgt
    werden, sobald die Anfrage (inkl. gestreamter Antwort) komplett fertig
    ist - siehe main.py proxy_v1()/gen()."""
    global _active, _last_arrival
    _ensure_dispatcher()
    async with _get_lock():
        _last_arrival = time.monotonic()
        cfg = get_config()
        limit = max(1, cfg.max_concurrent_requests)
        if _active < limit and not _waiters:
            # Freier Slot UND niemand wartet bereits davor - direkt durch,
            # kein unnötiges Debounce für den Normalfall (Anzahl gleichzeitiger
            # Anfragen liegt unterhalb des Limits).
            _active += 1
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        _waiters[rid] = fut
    telemetry.mark_queued(rid)
    try:
        await fut
    except asyncio.CancelledError:
        # Manuell abgebrochen (siehe cancel_waiting(), Dashboard-Button) oder
        # der Client selbst hat die Verbindung/den Request-Task beendet,
        # während wir noch warteten - in beiden Fällen: aus der Warteschlange
        # raus (falls noch drin - cancel_waiting() hat sie ggf. schon entfernt)
        # und die Cancellation normal weiterreichen, NICHT als erfolgreich
        # übergeben behandeln.
        async with _get_lock():
            _waiters.pop(rid, None)
        raise
    telemetry.mark_dequeued(rid)


def release() -> None:
    """Gibt den durch acquire() belegten Slot wieder frei. Kein awaitbarer
    Aufruf nötig (reines Zähler-Dekrement) - der nächste Dispatcher-Tick
    (max. _POLL_INTERVAL später) übergibt bei Bedarf die nächste wartende
    Anfrage, sobald auch die Ruhephase erreicht ist."""
    global _active
    _active = max(0, _active - 1)


def cancel_waiting(rid: str) -> bool:
    """Bricht eine NOCH WARTENDE (nicht bereits übergebene) Anfrage ab -
    genutzt vom Dashboard-"Abbrechen"-Button, wenn der Request noch im Status
    "queued" hängt (siehe main.py /requests/{rid}/cancel: dort zuerst
    versucht, danach Fallback auf den bisherigen Abbruch für bereits
    übergebene/laufende Anfragen). False, falls rid nicht (mehr) wartet - z.B.
    weil der Dispatcher sie in der Zwischenzeit bereits übergeben hat."""
    fut = _waiters.pop(rid, None)
    if fut is None or fut.done():
        return False
    fut.cancel()
    return True


def queue_depth() -> int:
    return len(_waiters)
