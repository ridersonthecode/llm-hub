# vLLM-Manager – Installation & Benutzung

Ersetzt das alte `Anleitung.txt` / `setup_hermes_service.sh` / `getHuggingfaceModel.py`
Setup (jetzt unter [`archive/`](archive/)) durch einen **Ollama-artigen** vLLM-Betrieb:
ein Dienst läuft dauerhaft, Modelle werden bei Bedarf automatisch geladen, Downloads
laufen per HTTP mit Fortschrittsanzeige, und ein MCP-Server macht das Ganze für
KI-Agenten steuerbar.

## Architektur

```
vllm.service (systemd, läuft immer)
  = FastAPI "vLLM-Manager" + MCP-Server, ein Prozess, Port 11434, 0.0.0.0
      │
      ├─ /v1/*        OpenAI-kompatibler Proxy, lädt Modell automatisch nach
      ├─ /models       Übersicht: registriert / gecacht / geladen
      ├─ /models/pull   Download starten (Hintergrund-Job)
      ├─ /models/pull/{job_id}  Fortschritt abfragen
      ├─ /models/{model}/load|unload  manuell steuern
      └─ /mcp          MCP-Server (Streamable HTTP) für KI-Zugriff
      │
      └─ spawnt bei Bedarf als Kindprozess (127.0.0.1:18811, intern):
         vllm serve <model> --gpu-memory-utilization ... --max-model-len ...
```

Nur **ein** Modell läuft gleichzeitig (unified memory ist zwischen CPU/GPU geteilt,
~121GB gesamt). Wird ein anderes Modell angefragt, wird das alte automatisch beendet
und das neue gestartet. Es gibt **kein** automatisches Idle-Entladen
(`idle_timeout_seconds: null` in der config.json) – ein Modell bleibt geladen, bis
ein anderes angefragt oder es manuell entladen wird.

## Dienst steuern

```bash
sudo systemctl status vllm
sudo systemctl restart vllm      # nur nötig bei Config-/Code-Änderungen am Manager selbst,
                                  # NICHT zum Modellwechsel!
sudo systemctl stop vllm
sudo systemctl enable/disable vllm
sudo journalctl -u vllm -f       # Manager-Logs (Start/Stop/Requests)
```

Logs der jeweils laufenden vLLM-Engine (Gewichte laden, CUDA-Graphen, Fehler) liegen
unter `~/vllm/logs/<model>.log` (Pfad auch in `GET /health` → `log_file`).

## config.json

Zentrale, **user-eigene** Datei (`~/vllm/config.json`) – kein `sudo` zum Ändern nötig
(anders als vorher, wo die Modellwahl in der root-eigenen systemd-Unit stand).
Nach Änderungen: `sudo systemctl restart vllm` (nur der Manager, dauert Sekunden,
nicht Minuten wie ein Modell-Start).

Wichtige Felder:

| Feld | Bedeutung |
|---|---|
| `host` / `port` | Wo der Manager lauscht (aktuell `0.0.0.0:11434`, netzwerkweit erreichbar – **keine Firewall-Regel** aktiv, siehe Sicherheit unten) |
| `api_key.enabled` / `api_key.key` | API-Key-Pflicht an/aus. **Aktuell `false`** – Server ist ohne jeden Header nutzbar. Zum Aktivieren: `key` setzen, `enabled: true`, Dienst neu starten. Dann muss jeder Request `Authorization: Bearer <key>` mitschicken (außer `/health`). |
| `idle_timeout_seconds` | `null` = nie automatisch entladen. Zahl (Sekunden) = Ollama-artiges Verhalten. |
| `default_model` | Wird verwendet, wenn ein Request kein `"model"`-Feld mitschickt. |
| `default_serve_args` | Fallback-Werte für `gpu_memory_utilization` / `max_model_len`, falls ein Modell keine eigenen hat. |
| `models.<name>` | Pro-Modell-Overrides: `tool_call_parser`, `max_model_len`, `gpu_memory_utilization`, `enable_auto_tool_choice`, `extra_args` (Liste beliebiger zusätzlicher `vllm serve`-Flags), `hf_token`, `enabled` (auf `false` setzen um ein kaputtes/gesperrtes Modell zu deaktivieren, ohne den Eintrag zu löschen), `notes`. |

## API benutzen

Genau wie vorher OpenAI-kompatibel, **aber ohne dass vorher ein bestimmtes Modell
manuell gestartet werden muss** – der erste Request an ein Modell lädt es automatisch
(dauert je nach Größe 1–5 Minuten, folgende Requests sind sofort schnell):

```bash
curl http://<LAN-IP>:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "Hallo, wer bist du?"}]
      }'
```

Verfügbare Modelle:

```bash
curl http://<LAN-IP>:11434/models          # Management-Sicht: registriert/gecacht/geladen
curl http://<LAN-IP>:11434/v1/models       # OpenAI-kompatible Liste
```

## Modelle herunterladen (mit Fortschritt)

```bash
# Download anstoßen
curl -X POST http://<LAN-IP>:11434/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-8B"}'
# -> {"job_id": "abcd1234ef56"}

# Fortschritt abfragen
curl http://<LAN-IP>:11434/models/pull/abcd1234ef56
# -> {"state":"downloading","bytes_total":...,"bytes_done":...,"percent":42.3,
#     "speed_mbps":38.1,"eta_seconds":210, ...}

# Alle Jobs (auch abgeschlossene) auflisten
curl http://<LAN-IP>:11434/models/pull
```

Für gated Modelle (z.B. Meta Llama): `"hf_token": "hf_xxx"` im Request-Body mitgeben,
oder dauerhaft in `config.json` unter dem Modell-Eintrag (`hf_token`) hinterlegen.

**Hinweis:** Download-Jobs laufen im Manager-Prozess und überleben keinen
`systemctl restart vllm` – bei einem Neustart während eines laufenden Downloads muss
er neu gestartet werden (bereits heruntergeladene Dateien bleiben im Cache erhalten,
`snapshot_download` setzt fort statt neu zu beginnen).

## Manuell laden/entladen

```bash
curl -X POST http://<LAN-IP>:11434/models/Qwen%2FQwen3-8B/load     # blockiert bis bereit
curl -X POST http://<LAN-IP>:11434/models/Qwen%2FQwen3-8B/unload
```

## Live-Dashboard

Erreichbar unter `http://<LAN-IP>:11434/dashboard` (z.B. `http://10.7.21.3:11434/dashboard`).
Zeigt in Echtzeit (Server-Sent Events, kein Neuladen der Seite nötig):

- Geladenes Modell + Laufzeit
- Zeitpunkt des letzten Prompts ("vor Xs")
- Requests laufend/wartend, KV-Cache-Auslastung, Ø TTFT / Ø ms-pro-Token (direkt aus
  vLLMs eigenem `/metrics`-Endpoint, also die "echten" Engine-Werte)
- Aktive Anfrage: Modell-Ladezeit (falls gerade ein Kaltstart lief) getrennt von der
  reinen Generierungs-TTFT, sowie Tokens grob live mitgezählt
- Laufende Downloads (gleiche Daten wie `/models/pull`)
- Tabelle der letzten ~30 Anfragen mit Dauer, TTFT, Prompt-/Completion-Tokens

**Technik:** `GET /dashboard/events` liefert einen SSE-Stream, der bei jeder
Zustandsänderung sofort pusht (neuer Request, Modellwechsel) und zusätzlich jede
Sekunde einen Heartbeat sendet (damit auch reine Engine-Metriken wie
KV-Cache-Auslastung aktuell bleiben, auch ohne neue Anfragen).

**Caveats:**
- Token-Zählung pro aktivem Request ist eine Näherung (1 SSE-Chunk mit Content ≈ 1
  Token) – exakte `prompt_tokens`/`completion_tokens` gibt es nur, wenn der Client
  bei Streaming-Requests `stream_options: {"include_usage": true}` mitschickt, oder
  bei Nicht-Streaming-Requests (dort immer exakt aus der Response geparst).
- Download-Geschwindigkeit kann bei vielen gleichzeitig laufenden Downloads
  zwischen zwei 2-Sekunden-Messungen kurzzeitig `0 MB/s` anzeigen (Bandbreite wird
  in Schüben zwischen den Jobs aufgeteilt) – die Byte-/Prozent-Werte sind trotzdem
  immer korrekt.
- Bei aktiviertem `api_key`: `/dashboard` selbst ist frei erreichbar (zeigt sonst
  nur eine leere Seite mit Passwort-Prompt), aber `/dashboard/status` und
  `/dashboard/events` verlangen denselben Bearer-Token wie die restliche API. Die
  Seite fragt den Key beim ersten Laden per Prompt ab und merkt ihn sich für die
  Browser-Session (`sessionStorage`).

### Verfügbare Modelle (2-spaltige Liste + Klick-Modal)

Ganz unten im Dashboard: alle registrierten (`config.json`) und zusätzlich lokal
gecachten Modelle, mit Badges (gecacht/geladen/deaktiviert/Vision). Klick auf ein
Modell öffnet ein Modal mit:
- Link zur HuggingFace-Seite (`https://huggingface.co/<model>`)
- fertigem JSON-Schnipsel im `chatLanguageModels.json`-Format (siehe VS-Code-Setup
  weiter oben) zum direkten Einfügen ins `"models"`-Array eines
  `customendpoint`-Eintrags - mit Kopieren-Button. Die `url` wird dynamisch aus der
  Adresse gebaut, über die das Dashboard gerade aufgerufen wurde.

## MCP-Server (für KI-Agenten)

Erreichbar unter `http://<LAN-IP>:11434/mcp` (Streamable-HTTP-Transport). Werkzeuge:

- `list_models` – registrierte, gecachte und aktuell geladene Modelle
- `server_status` – Status der aktiven Engine
- `pull_model(model, revision?, hf_token?)` – Download starten, gibt `job_id` zurück
- `pull_status(job_id)` – Fortschritt abfragen
- `load_model(model)` – Modell laden (blockierend)
- `unload_model()` – aktuelles Modell entladen

Einen MCP-Client (z.B. Claude, Hermes Agent) einfach auf diese URL zeigen lassen.

## Von Ollama auf vLLM übernommene Modelle

Ollama wurde gestoppt und deaktiviert (`systemctl disable ollama`). Alle zuvor über
Ollama genutzten Modelle wurden auf ihr HuggingFace-Äquivalent gemappt und in
`config.json` registriert (Downloads liefen am 2026-08-22 an, Fortschritt s.o.):

| Ollama-Modell | vLLM/HuggingFace-Äquivalent | Größe |
|---|---|---|
| `qwen3:8b` | `Qwen/Qwen3-8B` | 16.4 GB |
| `nemotron-3-nano:4b` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` | 5.3 GB |
| `qwen3.8:27b` | `Qwen/Qwen3.8-27B-FP8` | 30.9 GB |
| `qwen3-coder:30b-a3b-q4_K_M` + `qwen3-coder-256k:latest` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 31.2 GB |
| `qwen3.6:35b` | `Qwen/Qwen3.6-35B-A3B-FP8` | 37.5 GB |
| `nemotron-3.5-lightning:latest` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 21.6 GB |
| `nemotron-cascade-2:latest` | `nvidia/Nemotron-Cascade-2-30B-A3B` | 63.2 GB (nur BF16 verfügbar) |
| `qwen3-next:80b` | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 50.8 GB |
| `llama3.2:latest` | `meta-llama/Llama-3.2-3B-Instruct` | 6.4 GB | **gated – noch nicht heruntergeladen, siehe unten** |

**Wichtige Einschränkung:** Alle Tool-Call-Parser für die Nemotron-Modelle und
`llama3_json` für Llama wurden **nicht live getestet** (nur `hermes` für
Hermes-3-8B ist aus dem alten Setup bekannt funktionierend). Falls Tool-Calling bei
einem der neuen Modelle Fehler wirft, in `config.json` den `tool_call_parser`
anpassen (verfügbare Parser: `vllm serve --help` bzw.
`vllm/tool_parsers/__init__.py` im venv).

Ebenfalls unverändert übernommen bzw. bereits lokal vorhanden (nicht Teil der
Ollama-Migration, aber weiterhin nutzbar): `NousResearch/Hermes-3-Llama-3.1-8B`
(Default-Modell), `NousResearch/Hermes-4-14B`, `NousResearch/Hermes-4-70B`,
`Qwen/Qwen2.5-0.5B-Instruct` (aus `GET /models` ersichtlich, aktuell nicht in
`config.json` registriert – Requests dorthin funktionieren trotzdem, nur ohne
Custom-Parameter).

### Llama-3.2 nachträglich aktivieren

1. Auf huggingface.co einloggen und die Lizenz unter
   `meta-llama/Llama-3.2-3B-Instruct` akzeptieren.
2. In `config.json` beim Eintrag `meta-llama/Llama-3.2-3B-Instruct`: `hf_token` auf
   den eigenen HF-Token setzen, `enabled: true`.
3. Download anstoßen: `POST /models/pull {"model": "meta-llama/Llama-3.2-3B-Instruct"}`.

### gpt-oss-120b

Weiterhin als **deaktiviert** (`enabled: false`) in `config.json` hinterlegt –
bekannter aarch64-Bug in `openai-harmony` (siehe altes `archive/`-Anleitung bzw.
github.com/openai/harmony/issues/46). Sobald ein Fix existiert: `enabled: true`
setzen und neu starten.

## Sicherheit

- **Netzwerk:** `ufw` ist auf diesem System komplett inaktiv, der Manager bindet auf
  `0.0.0.0:11434` → **erreichbar für jeden mit Route zu dieser Maschine**, nicht nur
  das LAN-Subnetz. Das war eine explizite Entscheidung (kein Firewall-Scoping mehr
  wie in der alten `ufw`-Regel für `10.7.0.0/21`). Falls das doch eingeschränkt
  werden soll: `sudo ufw allow from 10.7.0.0/21 to any port 11434 proto tcp` und
  `sudo ufw enable`.
- **API-Key:** aktuell deaktiviert (siehe oben) – bewusst, damit der Server ohne
  Zusatzaufwand nutzbar ist. Der Key liegt (wenn gesetzt) im Klartext in
  `config.json` (Datei ist aber user-eigen, nicht root-eigen wie vorher die
  systemd-Unit) – zum Aktivieren siehe `config.json`-Tabelle oben.
- Anders als vorher taucht der API-Key **nicht mehr** in der Prozessliste
  (`ps aux`) auf – die interne `vllm serve`-Engine läuft ohne `--api-key`, nur
  über localhost (`127.0.0.1:18811`), die Authentifizierung passiert ausschließlich
  im Manager.

## Neues Modell hinzufügen (nicht aus obiger Liste)

In `config.json` unter `"models"` einen neuen Eintrag anlegen (siehe bestehende als
Vorlage), dann per `POST /models/pull` herunterladen und per `POST /v1/chat/completions`
bzw. `POST /models/{model}/load` verwenden – kein Dienst-Neustart nötig, `config.json`
wird bei jedem `ensure_loaded()`-Aufruf frisch aus dem geladenen Config-Objekt
gelesen (Neustart des Manager-Dienstes nur nötig, wenn `config.json` sich seit dem
letzten Start geändert hat).

## Manueller Start (zum Debuggen, ohne Dienst)

```bash
source ~/vllm/.venv/bin/activate
python -m vllm_manager
```

## Projektstruktur

```
~/vllm/
  config.json              # zentrale Konfiguration
  vllm.service              # systemd-Unit-Vorlage (installiert unter /etc/systemd/system/)
  vllm_manager/              # FastAPI + MCP Code
  logs/                      # Log-Dateien je Modell (vllm serve stdout/stderr)
  models/                    # HF_HOME, HuggingFace-Modell-Cache
  archive/                   # altes Setup (setup_hermes_service.sh, getHuggingfaceModel.py)
```
