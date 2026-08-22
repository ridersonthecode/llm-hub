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
      └─ spawnt bei Bedarf einen oder mehrere Kindprozesse (127.0.0.1:18811, 18812, ...):
         vllm serve <model> --gpu-memory-utilization ... --max-model-len ...
```

### Hot Pool: mehrere Modelle gleichzeitig warmhalten

`max_concurrent_models` in `config.json` (Default `2`) legt fest, wie viele
`vllm serve`-Kindprozesse gleichzeitig laufen dürfen – jeder auf einem eigenen
Port (`engine_port`, `engine_port+1`, ...). Solange ein angefragtes Modell
bereits im Pool ist und bereit steht, ist der Wechsel **instant** (kein
Kaltstart) – der Proxy routet den Request einfach an den passenden Port.

Da die GB10 Unified Memory hat (CPU und GPU teilen sich denselben Speicher,
~121GB gesamt) und jede Engine ihren Speicheranteil unabhängig von den anderen
reserviert, gibt es zusätzlich `gpu_memory_ceiling` (Default `0.9`) als
Sicherheitsnetz: die Summe der `gpu_memory_utilization` aller gleichzeitig
laufenden Engines darf diesen Wert nicht überschreiten. Passt ein neu
angefragtes Modell nicht mehr rein (Poolgröße voll ODER Speicherbudget
überschritten), wird automatisch die am längsten ungenutzte Engine verdrängt
(LRU) – Engines mit gerade aktiver Anfrage werden dabei nach Möglichkeit
übersprungen. Bei `max_concurrent_models: 1` verhält sich der Manager wie vor
diesem Feature: exklusiv, jeder Wechsel ist ein Kaltstart.

**Praktisch heißt das:** zwei kleinere Modelle (z.B. `Qwen3-8B` +
`NVIDIA-Nemotron-3-Nano-4B-FP8`, zusammen ~0.65 Speicherbudget) bleiben
problemlos parallel geladen. Bei den großen 30B–80B-Modellen (0.5–0.7 Budget
pro Modell) reduziert sich das Verhalten je nach Kombination automatisch
wieder auf "immer nur eins" – das Sicherheitsnetz verhindert OOM, ohne dass
man die Kombinationen von Hand ausschließen muss.

Es gibt **kein** automatisches Idle-Entladen (`idle_timeout_seconds: null` in
der config.json) – ein Modell bleibt geladen, bis es verdrängt, ein anderes
Modell dasselbe Slot anfragt oder es manuell entladen wird.

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
| `max_concurrent_models` | Größe des Hot Pools – so viele Modelle können gleichzeitig geladen bleiben (siehe Architektur oben). Default `2`. |
| `gpu_memory_ceiling` | Obergrenze für die Summe der `gpu_memory_utilization` aller gleichzeitig laufenden Engines. Default `0.9`. |
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

## Ollama-Kompatibilität (für alte, gegen Ollama gebaute Tools)

Eigene Skripte/Tools, die noch direkt gegen Ollamas native API sprechen (z.B.
`POST {base_url}/api/chat` statt `/v1/chat/completions`), müssen **nicht**
angepasst werden – der Manager übersetzt das automatisch:

- `POST /api/chat`: nimmt Ollamas Request-Format entgegen (`model`, `messages`,
  `format` als JSON-Schema, `think`, `options.temperature/top_p/num_predict`,
  `stream`) und übersetzt es auf `/v1/chat/completions` (inkl. `format` →
  `response_format: {type: json_schema, ...}` für strukturierten Output,
  `think` → `chat_template_kwargs.enable_thinking`). Antwortet im
  Ollama-Format (`{"message": {"content": ...}, "done": true, ...}`).
  **Nur `"stream": false`** wird unterstützt (Ollamas NDJSON-Streaming-Format
  ist nicht implementiert) – der Ollama-Default, und was reale Alt-Clients
  bisher genutzt haben.
- `GET /api/tags`: listet verfügbare Modelle im Ollama-Format.
- Alte Ollama-Modellnamen (`qwen3.8:27b` usw.) werden automatisch auf die neuen
  HuggingFace-Namen gemappt (Tabelle siehe unten) – man kann in der Config des
  Alt-Tools also einfach den bisherigen `model`-Namen stehen lassen.
- `keep_alive` wird entgegengenommen, aber ignoriert – das Idle-Verhalten wird
  hier zentral über `idle_timeout_seconds` bzw. den Hot Pool gesteuert, nicht
  pro Request.
- Andere Ollama-Endpunkte (`/api/generate`, `/api/pull`, `/api/show`, ...) gibt
  es nicht – bei Bedarf in [`vllm_manager/ollama_compat.py`](vllm_manager/ollama_compat.py)
  nach demselben Muster ergänzen.

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
Zeigt in Echtzeit per WebSocket (kein Polling, kein Neuladen der Seite nötig):

- **Geladene Modelle** (Hot Pool, siehe Architektur oben) als Tabelle: Modell, Status
  (🥶 Kaltstart / ✅ bereit), Port, Laufzeit, Requests laufend/wartend, KV-Cache-
  Auslastung, Ø TTFT / Ø ms-pro-Token, Tokens, und ein **"Entladen"-Button** je Zeile
  zum manuellen Stoppen eines Modells (ruft `POST /models/{model}/unload` auf, nach
  Bestätigung) – je Eintrag mit Live-Metriken direkt aus vLLMs eigenem
  `/metrics`-Endpoint der jeweiligen Engine
- **Pool-Speicherbudget**: Summe der `gpu_memory_utilization` aller geladenen Engines
  gegen `gpu_memory_ceiling`, als Balken
- Zeitpunkt des letzten Prompts ("vor Xs")
- Aktive Anfrage: Modell-Ladezeit (falls gerade ein Kaltstart lief) getrennt von der
  reinen Generierungs-TTFT, sowie Tokens grob live mitgezählt
- System-Auslastung: GPU-Auslastung/-Temperatur/-Leistung (`nvidia-smi`) und
  RAM-Auslastung (`/proc/meminfo`, deckt wegen Unified Memory auch den
  GPU-Speicherbedarf ab) als Live-Charts
- Modell-Verlauf (wie `ollama ps`, aber historisch): welches Modell wann geladen/
  entladen/verdrängt/abgestürzt ist, inkl. Grund
- Laufende Downloads (gleiche Daten wie `/models/pull`), ganz unten auf der Seite
- Tabelle der letzten ~30 Anfragen mit Dauer, TTFT, Prompt-/Completion-Tokens
- Dark-Mode-Toggle oben rechts (merkt sich die Wahl in `localStorage`)
- Sprachauswahl oben rechts (Dropdown): **Englisch ist Standardsprache**, Deutsch
  wählbar, Wahl wird persistent in `localStorage` gespeichert. Übersetzungen liegen
  in [`vllm_manager/languages/`](vllm_manager/languages/) – je eine `<code>.json`
  pro Sprache (aktuell `en.json`, `de.json`), automatisch beim Serverstart geladen
  und in die Seite eingebettet. Neue Sprache hinzufügen: einfach eine weitere
  `<code>.json` mit denselben Schlüsseln in den Ordner legen, kein Code-Änderung
  nötig – taucht dann automatisch im Dropdown auf.

**Technik:** `GET /dashboard/ws` (WebSocket) pusht bei jeder Zustandsänderung sofort
(neuer Request, Modellwechsel, Verdrängung) und zusätzlich jede Sekunde einen
Heartbeat (damit auch reine Engine-/System-Metriken aktuell bleiben, auch ohne neue
Anfragen). `GET /dashboard/status` liefert denselben Snapshot einmalig als JSON, z.B.
zum Debuggen mit `curl`.

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
  `/dashboard/ws` verlangen denselben Bearer-Token wie die restliche API (bei der
  WebSocket-Verbindung als erste Nachricht nach dem Verbindungsaufbau). Die Seite
  fragt den Key beim ersten Laden per Prompt ab und merkt ihn sich für die
  Browser-Session (`sessionStorage`).

### Verfügbare Modelle (2-spaltige Liste + Klick-Modal)

Alle registrierten (`config.json`) und zusätzlich lokal gecachten Modelle, mit Badges
(gecacht/geladen/deaktiviert/Vision). Klick auf ein Modell öffnet ein Modal mit:
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
