#!/usr/bin/env python3
"""Lädt ein HuggingFace-Modell manuell in den vLLM Cache herunter und
schaltet den laufenden llm-hub.service standardmäßig auf dieses Modell um.

Nutzung:
    python getHuggingfaceModel.py <modelname> [--revision REVISION] [--no-reload]

Beispiel:
    python getHuggingfaceModel.py NousResearch/Hermes-3-Llama-3.1-8B

Das Umschalten des Dienstes ruft intern 'sudo' auf (Passwortabfrage im
Terminal ist normal) und ersetzt nur den Modellnamen hinter 'vllm serve'
in /etc/systemd/system/llm-hub.service. Andere Flags (z.B. --tool-call-parser
hermes) bleiben unverändert und sollten bei nicht-Hermes-Modellen ggf.
manuell angepasst werden.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("HF_HOME", "/home/mwagner/llm-hub/models")

from huggingface_hub import snapshot_download

SERVICE_PATH = "/etc/systemd/system/llm-hub.service"
SERVICE_NAME = "llm-hub.service"


def reload_vllm_service(model_name: str) -> bool:
    print(f"Schalte {SERVICE_NAME} auf Modell '{model_name}' um (sudo erforderlich) ...")

    try:
        content = subprocess.run(
            ["sudo", "cat", SERVICE_PATH], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"Konnte {SERVICE_PATH} nicht lesen: {e.stderr}", file=sys.stderr)
        return False

    new_content, count = re.subn(
        r"(ExecStart=\S+vllm serve )\S+",
        lambda m: m.group(1) + model_name,
        content,
    )
    if count == 0:
        print(
            f"Konnte 'vllm serve <modell>' nicht in {SERVICE_PATH} finden — "
            "Dienst wurde nicht verändert.",
            file=sys.stderr,
        )
        return False

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".service") as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name

        subprocess.run(["sudo", "cp", tmp_path, SERVICE_PATH], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "restart", SERVICE_NAME], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Neustarten von {SERVICE_NAME}: {e}", file=sys.stderr)
        return False
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    print(
        f"{SERVICE_NAME} neu gestartet mit '{model_name}'. "
        f"Fortschritt: journalctl -u vllm -f"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Modell herunterladen")
    parser.add_argument("modelname", help="z.B. NousResearch/Hermes-3-Llama-3.1-8B")
    parser.add_argument("--revision", default=None, help="Branch/Tag/Commit (optional)")
    parser.add_argument("--token", default=None, help="HuggingFace Token für gated Modelle (optional)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Nur herunterladen, llm-hub.service nicht automatisch umschalten/neustarten",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Nur lokalen Cache verwenden, kein Netzwerk-Check (schneller Wechsel "
        "zwischen bereits heruntergeladenen Modellen)",
    )
    args = parser.parse_args()

    print(f"Lade Modell '{args.modelname}' nach {os.environ['HF_HOME']} ...")

    try:
        path = snapshot_download(
            repo_id=args.modelname,
            revision=args.revision,
            token=args.token,
            local_files_only=args.offline,
        )
    except Exception as e:
        print(f"Fehler beim Download: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Fertig. Modell liegt unter: {path}")

    if not args.no_reload:
        if not reload_vllm_service(args.modelname):
            sys.exit(1)


if __name__ == "__main__":
    main()
