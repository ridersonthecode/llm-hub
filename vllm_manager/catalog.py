"""Scannt den lokalen HuggingFace-Cache (hf_home) nach bereits heruntergeladenen Modellen."""
from __future__ import annotations

from pathlib import Path


def list_cached_models(hf_home: str) -> list[str]:
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
