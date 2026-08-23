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
import os
import shutil
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


# --- Lokale Dateien eines Modells löschen (Dashboard "Von Platte löschen") ---
# Ergänzt den reinen Read-Only-Scan oben um die Kehrseite: ein aus config.json
# entferntes oder nie registriertes, aber lokal gecachtes Modell nimmt sonst
# unbegrenzt Platz weg. Bewusst NICHT automatisch an "Remove" im Config-Editor
# gekoppelt (siehe Anleitung.md) - das ist ein eigener, expliziter Klick.

def cache_dir_for(model: str, hf_home: str) -> Path:
    """Pfad zum lokalen Download-Verzeichnis eines Modells.

    Zwei Fälle, analog zu downloader._cache_dir_for: normale HuggingFace-
    Modelle liegen unter hf_home/hub im "models--<org>--<name>"-Konventions-
    Verzeichnis; eigene lokale Modelle (z.B. selbst quantisierte AWQ-Varianten
    unter models-quantized/) sind in config.json bereits als absoluter Pfad
    hinterlegt - dort IST der Modellname der Pfad."""
    if model.startswith("/"):
        return Path(model)
    return Path(hf_home) / "hub" / ("models--" + model.replace("/", "--"))


def dir_size_bytes(path: Path) -> int:
    """Rekursive Verzeichnisgröße in Bytes. Blockierend (os.walk über
    potenziell hunderttausende Blob-/Snapshot-Dateien bei großen Modellen) -
    nur über asyncio.to_thread aufrufen, nie direkt im Event-Loop."""
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def delete_model_cache(model: str, hf_home: str) -> int:
    """Löscht das lokale Download-Verzeichnis eines Modells UNWIDERRUFLICH von
    der Platte. Blockierend (shutil.rmtree) - nur über asyncio.to_thread
    aufrufen. Gibt die Anzahl freigegebener Bytes zurück, 0 wenn es dort
    ohnehin nichts zu löschen gab. Prüft NICHT, ob das Modell gerade geladen
    ist - das muss der Aufrufer vorher sicherstellen (siehe main.py)."""
    path = cache_dir_for(model, hf_home)
    if not path.exists():
        return 0
    size = dir_size_bytes(path)
    shutil.rmtree(path, ignore_errors=False)
    return size


# --- Belegter Speicherplatz pro Modell (Dashboard-Anzeige "X GB") -----------
# dir_size_bytes() oben macht ein rekursives os.walk() - bei großen Modellen
# (hunderttausende Blob-/Snapshot-Dateien, teils >100GB) spürbar langsam.
# _models_catalog() (dashboard.py) ruft das aber für JEDES bekannte Modell bei
# JEDEM WebSocket-Heartbeat (1x/Sekunde und offenem Tab) ab - ungecacht würde
# das denselben Event-Loop-Stau riskieren wie der Scan oben (siehe
# Modul-Docstring). Verzeichnisgrößen ändern sich ohnehin nur nach einem
# abgeschlossenen Download oder Löschen, ein länger lebender Cache ist also
# unkritisch und wird an genau diesen Stellen aktiv invalidiert
# (downloader.py, main.py DELETE .../cache).
_SIZE_CACHE_TTL = 300.0
_size_cache: dict[str, int] = {}
_size_cache_at: dict[str, float] = {}
_size_locks: dict[str, asyncio.Lock] = {}


async def get_cached_size_bytes(model: str, hf_home: str) -> int:
    now = time.time()
    cached = _size_cache.get(model)
    if cached is not None and (now - _size_cache_at.get(model, 0)) < _SIZE_CACHE_TTL:
        return cached
    lock = _size_locks.setdefault(model, asyncio.Lock())
    async with lock:
        now = time.time()
        cached = _size_cache.get(model)
        if cached is not None and (now - _size_cache_at.get(model, 0)) < _SIZE_CACHE_TTL:
            return cached
        size = await asyncio.to_thread(dir_size_bytes, cache_dir_for(model, hf_home))
        _size_cache[model] = size
        _size_cache_at[model] = time.time()
        return size


def invalidate_size_cache(model: Optional[str] = None) -> None:
    """Nach einem abgeschlossenen Download oder Löschen aufrufen, damit die
    nächste Dashboard-Abfrage die echte (neue) Größe zurückgibt statt bis zu
    _SIZE_CACHE_TTL Sekunden lang die alte."""
    if model is None:
        _size_cache.clear()
        _size_cache_at.clear()
    else:
        _size_cache.pop(model, None)
        _size_cache_at.pop(model, None)
