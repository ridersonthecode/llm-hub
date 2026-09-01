# llm-hub

Ollama-artige Betriebsweise für [vLLM](https://github.com/vllm-project/vllm): ein
FastAPI-Dienst (`llm_hub`) läuft dauerhaft im Hintergrund, lädt Modelle beim
ersten Request automatisch (Kaltstart), bietet HTTP-Downloads mit
Fortschritts-Telemetrie, einen MCP-Server für KI-Agenten und ein Live-Dashboard
per WebSocket.

Vollständige Doku (Endpoints, config.json-Referenz, Sicherheit, Modell-Mapping,
Dashboard, MCP): siehe [Anleitung.md](Anleitung.md).

Frischer Rechner (z.B. ein zweiter DGX Spark)? Schritt-für-Schritt-Checkliste
von 0 auf laufenden Dienst inkl. RAG/Qdrant: siehe [INSTALL.md](INSTALL.md).

## Features

- **Auto-Load / Auto-Swap**: `vllm serve <model>` startet automatisch als
  Kindprozess bei der ersten Anfrage an ein Modell, kein manueller Dienst-Neustart
  nötig.
- **Hot Pool**: bis zu `max_concurrent_models` Modelle bleiben gleichzeitig geladen
  (je ein Kindprozess auf eigenem Port) – Wechsel zwischen bereits warmen Modellen
  ist instant statt eines Kaltstarts. Speicherbudget (`gpu_memory_ceiling`) und
  Poolgröße werden per LRU-Verdrängung automatisch eingehalten.
- **Automatisches Nachladen nach Neustart**: das zuletzt genutzte Modell wird bei
  jedem Dienststart im Hintergrund automatisch wieder geladen (abschaltbar über
  `auto_reload_last_model`), ohne den Serverstart zu blockieren.
- **OpenAI-kompatibler Proxy** unter `/v1/*`
- **Ollama-Kompatibilität** unter `/api/chat` + `/api/tags` – alte, gegen Ollama
  gebaute Tools laufen unverändert weiter (inkl. Mapping alter Modellnamen)
- **Downloads mit Live-Fortschritt**: `POST /models/pull`, Fortschritt über
  `GET /models/pull/{job_id}` (Bytes, Prozent, Geschwindigkeit, ETA)
- **MCP-Server** unter `/mcp` (Streamable HTTP) - Modelle per KI-Agent auflisten,
  herunterladen, laden/entladen
- **RAG** (optional): eigene Texte/Dokumente (PDF/TXT/MD) in Qdrant ablegen und
  per semantischer Suche wiederfinden - über `rag_*`-MCP-Tools, `/rag/*`-REST-API
  oder die Dashboard-Seite `/dashboard/rag`. Embedding-Modell läuft im Hot Pool
  wie jedes andere Modell.
- **Live-Dashboard** unter `/dashboard` (WebSocket, kein Polling): geladene
  Modelle inkl. Kaltstart-/Bereit-Status mit manuellem Entladen-Button, Modell-
  Verlauf, aktive Anfragen mit Endpoint/Port/TTFT/Live-Durchsatz je laufender
  Anfrage, GPU-/RAM-Live-Charts, Downloads, Modell-Katalog mit HF-Link +
  VS-Code-JSON-Snippet – zweisprachig (Englisch Standard, Deutsch wählbar,
  siehe [`llm_hub/languages/`](llm_hub/languages/))
- **Config-Editor** unter `/dashboard/config`: `config.json` per Formular
  bearbeiten (Inputs/Dropdowns/Checkboxen, inkl. Modell-Liste) statt roher
  JSON-Datei - mit serverseitiger Validierung, automatischem Backup vor jedem
  Speichern, Live-Übernahme ohne Neustart (bis auf `host`/`port`) oder
  optionalem vollem Dienst-Neustart, und Fallback auf die letzte funktionierende
  Config, falls `config.json` beim Start mal nicht lädt
- **Kostentracking** unter `/dashboard/costs` (rein fiktiv, der lokale Betrieb
  ist kostenlos): rechnet jede Anfrage gegen konfigurierbare $/MTok-Preise um
  (Default: Claude-Sonnet-5-Standardpreise) zum Vergleich "was hätte das über
  eine Cloud-API gekostet" - persistent, live per WebSocket, mit
  Einzel-/Mehrfach-Löschen und Reset
- **Optionaler API-Key** (standardmäßig deaktiviert), zentrale `config.json`
  (kein `sudo` zum Ändern nötig)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# "mcp[cli]<2" ist Pflicht, kein Nice-to-have: mcp 2.x hat FastMCP (siehe
# mcp_tools.py) ersatzlos in MCPServer umbenannt, der Dienst startet mit
# mcp>=2 gar nicht erst (live erlebt beim 2026-09-01-Umzug dieses Ordners).
pip install vllm fastapi "uvicorn[standard]" httpx huggingface_hub "mcp[cli]<2" qdrant-client pypdf

cp config.example.json config.json
# config.json anpassen: hf_home-Pfad, ggf. Modelle/Parameter

python -m llm_hub
```

Für RAG zusätzlich Qdrant starten (Docker) und in `config.json` unter `"rag"` aktivieren
– siehe [Anleitung.md](Anleitung.md#rag-retrieval-augmented-generation).

Für dauerhaften Betrieb: [llm-hub.service](llm-hub.service) nach
`/etc/systemd/system/llm-hub.service` kopieren (Pfade/User anpassen), dann:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llm-hub
```

## Projektstruktur

```
config.example.json   # Vorlage - kopieren nach config.json (gitignored)
llm_hub/          # FastAPI + MCP-Code
llm-hub.service           # systemd-Unit-Vorlage
Anleitung.md            # ausführliche Doku (Deutsch)
archive/                # Vorgänger-Setup (Referenz)
```

`config.json`, `models/` (HuggingFace-Cache) und `logs/` sind bewusst nicht
Teil des Repos (siehe `.gitignore`) - `config.json` kann API-Keys/HF-Tokens
enthalten, `models/` kann hunderte GB groß werden.
