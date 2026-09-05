"""Gemeinsame Registry laufender Upstream-Streams zur Engine (vLLM/SGLang).

Vorher lebte diese Registry nur als Modul-Variable in main.py und wurde
ausschließlich von proxy_v1()/gen() befüllt - der Ollama-Kompatibilitäts-
Layer (ollama_compat.py /api/chat) hatte dadurch GAR KEINEN Cancel-Pfad: der
dortige Engine-Aufruf war ein einziger blockierender client.post() ohne
jede Registrierung. Setzte der aufrufende Client (z.B. Python mit eigenem
Timeout) die Verbindung ab und gab auf, lief die Anfrage serverseitig
unbemerkt weiter, und der Dashboard-"Abbrechen"-Button lief immer in die 404
"Keine aktive, abbrechbare Anfrage ... gefunden" - siehe Chat vom 2026-09-01.

Ausgelagert in ein eigenes Modul (statt weiter nur in main.py), damit sowohl
main.py (proxy_v1/gen) als auch ollama_compat.py (api_chat) dieselbe Registry
nutzen können, ohne dass ollama_compat.py main.py importieren müsste
(main.py bindet ollama_compat.py bereits als Router ein - ein Import in die
Gegenrichtung wäre ein Zirkelimport).

Schließen der registrierten Upstream-Verbindung lässt die Engine einen
Client-Disconnect erkennen und die Generierung serverseitig beenden -
dieselbe Technik, die main.py schon beim automatischen Wiederholungsschleifen-
Abbruch nutzt (siehe main.py-Kommentare bei _has_repetition_loop), hier nur
von außen (manueller Cancel-Button) oder durch einen erkannten echten
Client-Disconnect (siehe watch_disconnect()) ausgelöst statt automatisch
durch die Loop-Erkennung."""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import Request

logger = logging.getLogger("llm_hub.active_streams")

# rid -> laufende Upstream-Response (httpx, stream=True) Richtung Engine.
_streams: dict[str, httpx.Response] = {}
# rids, deren Abbruch WIR ausgelöst haben (manueller Cancel-Button oder
# automatische Disconnect-Erkennung) - unterscheidet im except-Block der
# Aufrufer (main.py gen(), ollama_compat.api_chat()) einen absichtlichen
# Abbruch von einem echten Verbindungsfehler zur Engine.
_cancelled: set[str] = set()

# Wie oft watch_disconnect() den ASGI-Layer nach einem Client-Disconnect
# fragt - reines Polling auf bereits vorliegende Zustandsinfo (siehe
# Request.is_disconnected() Docstring), kein spürbarer Overhead, muss also
# nicht knapper sein als ein Mensch ohnehin auf einen abgebrochenen Request
# reagieren würde.
_DISCONNECT_POLL_INTERVAL = 2.0


def register(rid: str, upstream: httpx.Response) -> None:
    _streams[rid] = upstream


def unregister(rid: str) -> None:
    _streams.pop(rid, None)
    _cancelled.discard(rid)


def was_cancelled(rid: str) -> bool:
    return rid in _cancelled


async def cancel(rid: str) -> bool:
    """Bricht eine bereits an die Engine übergebene, gestreamte Anfrage ab
    (rid muss über register() bekannt sein - eine Anfrage, die noch in der
    Concurrency-Warteschlange wartet, siehe request_queue.cancel_waiting(),
    läuft über einen anderen Pfad). False, falls rid nicht (mehr) registriert
    ist - z.B. weil die Anfrage in der Zwischenzeit von selbst fertig wurde."""
    upstream = _streams.get(rid)
    if upstream is None:
        return False
    _cancelled.add(rid)
    await upstream.aclose()
    return True


async def watch_disconnect(request: Request, rid: str) -> None:
    """Läuft als Hintergrund-Task parallel zu einer laufenden Anfrage
    (main.py gen(), ollama_compat.api_chat()): bricht sie automatisch ab,
    sobald der ASGI-Server (uvicorn) einen ECHTEN TCP-Disconnect des Clients
    meldet - z.B. weil ein Python-Client mit eigenem Timeout die Verbindung
    schließt, während die Engine serverseitig unbemerkt weiter generiert.

    Bewusst NUR dieses harte Signal (Request.is_disconnected() liest die
    "http.disconnect"-Nachricht, die uvicorn beim tatsächlichen Schließen/
    Zurücksetzen des Sockets schickt) - kein Raten anhand von Wartezeit o.ä.
    Ein Client, der bei seinem Timeout NUR lokal aufgibt, den Socket aber
    (z.B. wegen Connection-Pooling in seiner HTTP-Bibliothek) nicht wirklich
    schließt, bleibt dadurch unentdeckt - dafür bleibt der manuelle
    Cancel-Button (cancel() oben) nötig, dieser Watcher deckt nur den Fall
    ab, in dem sicher feststeht, dass niemand mehr zuhört.

    Muss vom Aufrufer per .cancel() beendet werden, sobald die Anfrage selbst
    fertig ist (siehe finally-Blöcke der Aufrufer) - sonst liefe dieser Task
    ewig weiter."""
    try:
        while True:
            await asyncio.sleep(_DISCONNECT_POLL_INTERVAL)
            if await request.is_disconnected():
                logger.info(
                    "Client-Disconnect erkannt (Socket geschlossen), breche Anfrage automatisch ab (rid=%s)",
                    rid,
                )
                await cancel(rid)
                return
    except asyncio.CancelledError:
        pass
