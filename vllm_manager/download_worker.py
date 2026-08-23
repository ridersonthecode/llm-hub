"""Eigenständiger Download-Kindprozess für downloader.py.

Warum ein eigener Prozess statt eines Threads (wie zuvor via
loop.run_in_executor): huggingface_hubs snapshot_download() bietet keinerlei
Abbruch-Mechanismus, und ein bereits laufender Thread-Pool-Task lässt sich aus
Python heraus nicht mitten in der Ausführung stoppen (concurrent.futures kann
einen Future nur VOR dem Start canceln, nicht während er läuft). Ein Kind-
prozess dagegen lässt sich jederzeit sauber per SIGTERM/SIGKILL beenden -
genau das braucht der "Download abbrechen"-Button im Dashboard.

Aufruf (siehe downloader.py): python -m vllm_manager.download_worker <model>
  [--revision REV] --cache-dir DIR
HF-Token (falls gesetzt) kommt bewusst über die Umgebungsvariable HF_TOKEN,
nicht als Kommandozeilenargument - der volle Kommandozeilenaufruf ist über
'ps aux' für jeden auf dem Rechner sichtbar, ein Token darin wäre ein Leck."""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    try:
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            token=token,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        print(f"download_worker: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
