# vllm-manager

Ollama-artige Betriebsweise für [vLLM](https://github.com/vllm-project/vllm): ein
FastAPI-Dienst (`vllm_manager`) läuft dauerhaft im Hintergrund, lädt Modelle beim
ersten Request automatisch (Kaltstart), bietet HTTP-Downloads mit
Fortschritts-Telemetrie, einen MCP-Server für KI-Agenten und ein Live-Dashboard
per WebSocket.

Vollständige Doku (Endpoints, config.json-Referenz, Sicherheit, Modell-Mapping,
Dashboard, MCP): siehe [Anleitung.md](Anleitung.md).

## Features

- **Auto-Load / Auto-Swap**: `vllm serve <model>` startet automatisch als
  Kindprozess bei der ersten Anfrage an ein Modell, kein manueller Dienst-Neustart
  nötig.
- **Hot Pool**: bis zu `max_concurrent_models` Modelle bleiben gleichzeitig geladen
  (je ein Kindprozess auf eigenem Port) – Wechsel zwischen bereits warmen Modellen
  ist instant statt eines Kaltstarts. Speicherbudget (`gpu_memory_ceiling`) und
  Poolgröße werden per LRU-Verdrängung automatisch eingehalten.
- **OpenAI-kompatibler Proxy** unter `/v1/*`
- **Ollama-Kompatibilität** unter `/api/chat` + `/api/tags` – alte, gegen Ollama
  gebaute Tools laufen unverändert weiter (inkl. Mapping alter Modellnamen)
- **Downloads mit Live-Fortschritt**: `POST /models/pull`, Fortschritt über
  `GET /models/pull/{job_id}` (Bytes, Prozent, Geschwindigkeit, ETA)
- **MCP-Server** unter `/mcp` (Streamable HTTP) - Modelle per KI-Agent auflisten,
  herunterladen, laden/entladen
- **Live-Dashboard** unter `/dashboard` (WebSocket, kein Polling): geladene
  Modelle inkl. Kaltstart-/Bereit-Status mit manuellem Entladen-Button, Modell-
  Verlauf, aktive Anfrage mit TTFT/Tokenverbrauch, GPU-/RAM-Live-Charts, Downloads,
  Modell-Katalog mit HF-Link + VS-Code-JSON-Snippet – zweisprachig (Englisch
  Standard, Deutsch wählbar, siehe [`vllm_manager/languages/`](vllm_manager/languages/))
- **Optionaler API-Key** (standardmäßig deaktiviert), zentrale `config.json`
  (kein `sudo` zum Ändern nötig)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install vllm fastapi "uvicorn[standard]" httpx huggingface_hub "mcp[cli]"

cp config.example.json config.json
# config.json anpassen: hf_home-Pfad, ggf. Modelle/Parameter

python -m vllm_manager
```

Für dauerhaften Betrieb: [vllm.service](vllm.service) nach
`/etc/systemd/system/vllm.service` kopieren (Pfade/User anpassen), dann:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
```

## Projektstruktur

```
config.example.json   # Vorlage - kopieren nach config.json (gitignored)
vllm_manager/          # FastAPI + MCP-Code
vllm.service           # systemd-Unit-Vorlage
Anleitung.md            # ausführliche Doku (Deutsch)
archive/                # Vorgänger-Setup (Referenz)
```

`config.json`, `models/` (HuggingFace-Cache) und `logs/` sind bewusst nicht
Teil des Repos (siehe `.gitignore`) - `config.json` kann API-Keys/HF-Tokens
enthalten, `models/` kann hunderte GB groß werden.
