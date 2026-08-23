"""Scannt den lokalen HuggingFace-Cache (hf_home) nach bereits heruntergeladenen
Modellen.

Performance: dieser Scan macht synchrone Festplatten-I/O (iterdir/exists/glob
über potenziell viele Modell-Verzeichnisse) und wird an mehreren Stellen sehr
häufig aufgerufen - u.a. bei JEDER /v1/*-Chat-Anfrage (main.py: prüft, ob das
angefragte Modell bekannt ist) und bei JEDEM Dashboard-WebSocket-Heartbeat
(einmal pro Sekunde und offenem Tab). Ungecacht blockiert das kurz den
gesamten (single-threaded) asyncio-Event-Loop - bei Disk-I/O-Last (z.B.
während ein Modell lädt oder ein Download läuft) reicht das, um den
WebSocket-Ping/Pong-Keepalive des Dashboards zu verpassen ("Verbindung
verloren", erholt sich nach ein paar Sekunden von selbst) und im schlimmsten
Fall auch laufende Chat-Antworten kurz stocken zu lassen.

Fix: kurz gecacht (der Cache-Inhalt ändert sich ohnehin nur nach einem
abgeschlossenen Download oder manuellem Löschen aus dem Cache - eine Handvoll
Sekunden Verzögerung dabei sind unkritisch) UND der eigentliche Scan läuft bei
einem Cache-Miss in einem Thread (asyncio.to_thread), blockiert den Event-Loop
also selbst dann nicht, wenn der Scan mal langsamer ist."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

_CACHE_TTL = 3.0
_cache: dict[str, list[str]] = {}
_cache_at: dict[str, float] = {}
_lock = asyncio.Lock()


def _scan(hf_home: str) -> list[str]:
    """Der eigentliche (blockierende) Scan - nur über asyncio.to_thread aufrufen."""
    hub = Path(hf_home) / "hub"
    if not hub.exists():
        return []
    names = []
    for p in hub.iterdir():
        if not p.is_dir() or not p.name.startswith("models--"):
            continue
        # HF-Cache-Layout: "models--<org>--<name>" -> "<org>/<name>"
        parts = p.name[len("models--"):].split("--")
        # Snapshot muss existieren und Dateien enthalten, sonst zählt der Download nicht als vollständig
        snapshots = p / "snapshots"
        if not snapshots.exists() or not any(snapshots.iterdir()):
            continue
        # huggingface_hub legt während des Downloads "<hash>.incomplete"-Dateien im
        # blobs/-Verzeichnis an. Solange die existieren, ist der Download noch nicht
        # fertig - auch wenn schon einzelne (kleine) Dateien vollständig gelandet sind.
        blobs = p / "blobs"
        if blobs.exists() and any(blobs.glob("*.incomplete")):
            continue
        names.append("/".join(parts))
    return sorted(names)


async def list_cached_models(hf_home: str) -> list[str]:
    now = time.time()
    cached = _cache.get(hf_home)
    if cached is not None and (now - _cache_at.get(hf_home, 0)) < _CACHE_TTL:
        return cached
    async with _lock:
        # Zwischen dem ersten (ungelockten) Check oben und hier könnte ein
        # anderer Task den Cache schon aufgefrischt haben - erneut prüfen,
        # statt bei gleichzeitigen Anfragen mehrfach parallel zu scannen.
        now = time.time()
        cached = _cache.get(hf_home)
        if cached is not None and (now - _cache_at.get(hf_home, 0)) < _CACHE_TTL:
            return cached
        result = await asyncio.to_thread(_scan, hf_home)
        _cache[hf_home] = result
        _cache_at[hf_home] = time.time()
        return result


def invalidate_cache(hf_home: Optional[str] = None) -> None:
    """Erzwingt einen frischen Scan beim nächsten Aufruf - z.B. direkt nach
    einem abgeschlossenen Download, statt bis zu _CACHE_TTL Sekunden zu warten."""
    if hf_home is None:
        _cache.clear()
        _cache_at.clear()
    else:
        _cache.pop(hf_home, None)
        _cache_at.pop(hf_home, None)
