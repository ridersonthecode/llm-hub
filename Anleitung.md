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
(LRU). Bei `max_concurrent_models: 1` verhält sich der Manager wie vor diesem
Feature: exklusiv, jeder Wechsel ist ein Kaltstart.

**Engines mit einer gerade laufenden Anfrage werden NIE automatisch verdrängt**
(würde die Antwort mitten drin abbrechen). Reicht der Platz nur durch
Verdrängen einer beschäftigten Engine, schlägt das Laden des neuen Modells
stattdessen mit HTTP 503 fehl - die Fehlermeldung nennt das/die blockierende(n)
Modell(e) und verweist auf manuelles Entladen (Dashboard-Button oder
`POST /models/<model>/unload`). Sind mehrere Engines geladen und nur eine
davon ist gerade beschäftigt, wird ganz normal eine der freien verdrängt.

**Praktisch heißt das:** zwei kleinere Modelle (z.B. `Qwen3-8B` +
`NVIDIA-Nemotron-3-Nano-4B-FP8`, zusammen ~0.65 Speicherbudget) bleiben
problemlos parallel geladen. Bei den großen 30B–80B-Modellen (0.5–0.7 Budget
pro Modell) reduziert sich das Verhalten je nach Kombination automatisch
wieder auf "immer nur eins" – das Sicherheitsnetz verhindert OOM, ohne dass
man die Kombinationen von Hand ausschließen muss.

Es gibt **kein** automatisches Idle-Entladen (`idle_timeout_seconds: null` in
der config.json) – ein Modell bleibt geladen, bis es verdrängt, ein anderes
Modell dasselbe Slot anfragt oder es manuell entladen wird.

**Wichtiger Bugfix:** bis vor Kurzem hat ein einziger Kaltstart (egal welches
Modell) den kompletten Hot Pool faktisch blockiert - der interne Pool-Lock
(`process_manager._pool_lock`) hielt versehentlich auch die gesamte
Warte-auf-"gesund"-Schleife eines Kaltstarts (bis zu `startup_timeout_seconds`
lang), nicht nur die kurzen Setup-Schritte davor. Dadurch blockierte JEDER
andere `ensure_loaded()`-Aufruf - selbst für ein bereits fertig geladenes,
völlig anderes Modell (z.B. das RAG-Embedding-Modell beim Hinzufügen eines
Texts) - bis zu mehrere Minuten lang, nur weil irgendwo im Hintergrund ein
anderes Modell gerade kalt startete. Behoben: der Lock schützt jetzt nur noch
die kurze Prüfen/Verdrängen/Prozess-Start-Phase, die eigentliche Warteschleife
läuft danach außerhalb davon - mehrere gleichzeitige Anfragen für dasselbe,
noch ladende Modell warten dabei weiterhin korrekt auf denselben Kaltstart,
statt ihn versehentlich doppelt zu starten.

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
| `models.<name>` | Pro-Modell-Overrides: `tool_call_parser`, `reasoning_parser` (trennt `<think>...</think>` in ein eigenes `reasoning_content`-Feld statt es in die normale Antwort zu mischen, z.B. `"qwen3"` für Qwen3-Thinking-Modelle - gültige Werte: `vllm serve --help=all \| grep -A2 reasoning-parser`), `max_model_len`, `gpu_memory_utilization`, `enable_auto_tool_choice`, `extra_args` (Liste beliebiger zusätzlicher `vllm serve`-Flags), `hf_token`, `enabled` (auf `false` setzen um ein kaputtes/gesperrtes Modell zu deaktivieren, ohne den Eintrag zu löschen), `notes`. |
| _(alle obigen Felder auch per Formular statt von Hand)_ | Siehe [Config-Editor](#config-editor-dashboardconfig) unter Live-Dashboard weiter unten - Inputfelder/Dropdowns/Checkboxen statt rohem JSON, mit Validierung, automatischem Backup und Live-Übernahme oder Dienst-Neustart. |

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

**Download abbrechen:** per "Abbrechen"-Button in der Downloads-Tabelle im Dashboard,
oder `POST /models/pull/<job_id>/cancel`. Der eigentliche Transfer läuft in einem
eigenen Kindprozess (`download_worker.py`), der beim Abbrechen per SIGTERM beendet
wird (nach 5s Frist SIGKILL) - bereits heruntergeladene Teil-Dateien bleiben liegen
(fortsetzbar bei einem erneuten Download desselben Modells), werden aber NICHT
automatisch gelöscht. Wer den Platz sofort zurückwill: zusätzlich den
"Von Platte löschen"-Button nutzen (siehe unten, "Verfügbare Modelle").

Bewusst ein eigener Prozess statt eines Threads: `huggingface_hub`s
`snapshot_download()` bietet selbst keinen Abbruch-Mechanismus, und ein bereits
laufender Thread-Pool-Task lässt sich aus Python heraus nicht mitten in der
Ausführung stoppen (nur vor dem Start) - nur ein Kindprozess lässt sich jederzeit
sauber per Signal beenden.

**Hinweis:** Download-Jobs laufen im Manager-Prozess (der Kindprozess wird von ihm
verwaltet) und überleben keinen `systemctl restart vllm` – bei einem Neustart
während eines laufenden Downloads muss er neu gestartet werden (bereits
heruntergeladene Dateien bleiben im Cache erhalten, `snapshot_download` setzt fort
statt neu zu beginnen).

## Manuell laden/entladen

```bash
curl -X POST http://<LAN-IP>:11434/models/Qwen%2FQwen3-8B/load     # blockiert bis bereit
curl -X POST "http://<LAN-IP>:11434/models/Qwen%2FQwen3-8B/load?background=true"  # kehrt sofort zurück
curl -X POST http://<LAN-IP>:11434/models/Qwen%2FQwen3-8B/unload
```

Im Dashboard: **"Laden"-Button** im Klick-Modal jedes nicht geladenen, aktivierten
Modells (Gegenstück zum "Entladen"-Button) - ruft intern `?background=true` auf,
damit der Browser-Request bei großen Modellen nicht minutenlang hängt. Der
Kaltstart läuft serverseitig als überwachter Hintergrund-Task (exakt wie das
automatische Nachladen nach einem Neustart, siehe unten) - Fortschritt ("lädt
gerade (Kaltstart)") zeigt wie gewohnt die Tabelle "Geladene Modelle" bzw. der
Modell-Verlauf über den nächsten WebSocket-Heartbeat.

**Kaltstart-Timeout bei großen Modellen:** `startup_timeout_seconds` in
`config.json` (Default 1800s, Config-Editor unter "Hot Pool & Limits") begrenzt,
wie lange ein Kaltstart maximal dauern darf, bevor er als "Start-Timeout"
abgebrochen wird - reiner Schutz gegen einen hängenden/abgestürzten
Ladevorgang, kein künstliches Limit gegen große Modelle. Bei sehr großen
Modellen (kalter Seiten-Cache nach einem Neustart, viel gleichzeitige
Festplatten-Last durch andere Vorgänge) kann ein einzelner Kaltstart trotzdem
mal deutlich länger dauern als gewohnt - in dem Fall den Wert im Config-Editor
weiter erhöhen.

## Automatisches Nachladen nach Neustart

Ollama-artiges Verhalten (Default an, `auto_reload_last_model` in `config.json`
bzw. im Config-Editor unter "Hot Pool & Limits"): das zuletzt genutzte Modell
wird bei jedem Dienststart automatisch im Hintergrund nachgeladen, ohne den
Start des Managers selbst oder andere Requests zu blockieren.

- Der Zeiger aufs zuletzt genutzte Modell (`last_active_model.json` neben
  `config.json`, nicht Teil von Git) wird bei jeder Nutzung eines bereiten
  Modells aktualisiert - sowohl nach einem frischen Kaltstart als auch bei
  Wiederverwendung eines bereits warmen Modells. Bei mehreren gleichzeitig
  geladenen Modellen (Hot Pool) zählt jeweils das zuletzt tatsächlich genutzte.
- Ein **manuelles Entladen** (Dashboard-Button oder `POST .../unload`) löscht
  diesen Zeiger wieder - ein bewusst entladenes Modell soll nach einem
  Neustart nicht ungefragt zurückkommen. Automatische Vorgänge (Idle-Timeout,
  Verdrängung durch ein anderes Modell, Absturz) lassen den Zeiger dagegen
  unangetastet.
- Zum Deaktivieren: `"auto_reload_last_model": false` in `config.json`, oder
  die Checkbox im Config-Editor - dann startet der Hot Pool nach einem
  Neustart bewusst leer.

## Sicherheitsnetz gegen durchgehende Generierungen (`max_tokens` pro Modell)

Anlass: ein Modell ist in eine Wiederholungsschleife geraten und hat am Stück
über 80.000 Tokens produziert (43 Minuten lang), weil weder der Client noch
der Server eine Obergrenze gesetzt hatten - der einzige Client-seitige
Timeout hat nur "Sorry, no response was returned." angezeigt, während die
Engine im Hintergrund weiter generiert hat.

Pro Modell lässt sich in `config.json` (Feld `max_tokens` unter `models.<model>`)
bzw. im Config-Editor ("Max output tokens (safety net, empty = unbounded)")
eine Obergrenze für die Antwortlänge hinterlegen:

- Greift **nur**, wenn der Request selbst weder `max_tokens` noch
  `max_completion_tokens` mitschickt (OpenAI-API) bzw. kein `num_predict`
  (Ollama-Kompatibilität, `/api/chat`) - ein expliziter Client-Wunsch wird nie
  heruntergedrückt oder überschrieben.
- Gilt nur für `/v1/chat/completions`, `/v1/completions` und `/api/chat`.
  `/v1/embeddings` ist davon nie betroffen (dort gibt es kein `max_tokens`).
- `null`/leer = kein Limit (bisheriges Verhalten, nur durch `max_model_len`
  begrenzt) - so war jedes Modell vor dieser Änderung konfiguriert.

Die aktuell hinterlegten Werte sind bewusst großzügig gewählt (pro Modell
ca. ein Viertel des jeweiligen `max_model_len`, gedeckelt bei 65.536 Tokens
als Obergrenze, Boden bei 4.096) - sie sollen normale, auch sehr lange
Antworten nicht beschneiden, aber eine durchgehende Generierung auf ein
absehbares Maximum begrenzen. Feinjustierung pro Modell jederzeit über den
Config-Editor möglich.

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
- Aktive Anfragen als Tabelle (eine Zeile je laufender Anfrage - bei Hot Pool
  ggf. mehrere gleichzeitig): Modell, **App** (roher `User-Agent`-Header des
  aufrufenden Clients - "Unbekannt", falls keiner mitgeschickt wird; gekürzt
  mit vollem Wert im Tooltip - dieselbe Spalte auch bei Letzte Anfragen und auf
  der Kostenseite), Endpoint (`chat/completions`, `embeddings`, ...) inkl.
  Stream-Badge, Engine-Port, Status (🥶 Kaltstart falls gerade das Modell noch
  lädt / läuft), Gesamtlaufzeit, Modell-Ladezeit getrennt von der reinen
  Generierungs-TTFT, Tokens, Reasoning-Tokens (separat gezählt - Text aus
  `delta.reasoning_content` bei Modellen mit `reasoning_parser`, siehe
  config.json, zählt NICHT in die normale Tokens-Spalte hinein), und live
  berechneter Durchsatz (Tokens/Sek., bezieht sich nur auf die normale
  Antwort-Spalte, nicht auf Reasoning-Tokens)
- System-Auslastung: GPU-Auslastung/-Temperatur/-Leistung (`nvidia-smi`) und
  RAM-Auslastung (`/proc/meminfo`, deckt wegen Unified Memory auch den
  GPU-Speicherbedarf ab) als Live-Charts - direkt unter der Kachel-Übersicht,
  vor den modellbezogenen Abschnitten (Geladene Modelle/Aktive Anfragen/Verlauf
  bleiben dadurch zusammenhängend statt durch die Hardware-Anzeige getrennt)
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

Jede Tabelle wird bei jedem Tick nur dann im DOM ersetzt, wenn sich ihr Inhalt
tatsächlich geändert hat (`safeSetHTML()` in `dashboard.py`/`cost_dashboard.py`
vergleicht das neu gerenderte HTML gegen das zuletzt geschriebene) - vorher wurde
jede Tabelle jede Sekunde komplett neu aufgebaut, auch ohne Änderung, was aussah
wie ein ständiges Neuladen und dabei jede laufende Textmarkierung sofort wieder
aufgehoben hat. Zusätzlich wird ein Update verschoben, solange gerade Text
innerhalb der betroffenen Tabelle markiert ist - der nächste Tick (≤1s später)
holt es nach, sobald die Markierung beendet ist.

**Performance:** der Dienst läuft single-threaded auf einem asyncio-Event-Loop -
WebSocket-Heartbeat und Chat-Proxy teilen sich denselben Thread. Der Scan des
lokalen HF-Caches (`catalog.list_cached_models()`, u.a. für die
"Available Models"-Liste UND bei jeder `/v1/*`-Chat-Anfrage zur Modell-Validierung)
sowie `GET .../metrics`-Abrufe der laufenden Engine sind deshalb kurz gecacht
(3s bzw. 0.8s TTL, mit Lock-Dedup gegen mehrere gleichzeitig offene Dashboard-Tabs)
und laufen bei einem Cache-Miss in einem Thread statt den Event-Loop zu blockieren -
sonst kann kurzzeitige Disk-I/O-Last (Modell lädt, Download läuft) das
WebSocket-Ping/Pong-Keepalive verpassen lassen ("Verbindung unterbrochen", erholt
sich nach ein paar Sekunden von selbst) und im schlimmsten Fall auch laufende
Chat-Antworten kurz stocken lassen. Nach einem abgeschlossenen Download wird der
Cache sofort invalidiert statt bis zu 3s zu warten.

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
für gecacht/geladen/deaktiviert sowie die Fähigkeiten des Modells - Vision,
Tool Calling (`enable_auto_tool_choice`), Reasoning (`reasoning_parser` gesetzt)
und Embedding (`task: "embed"`) - auf einen Blick sichtbar, ohne erst ins Modal
klicken zu müssen. Bei lokal gecachten Modellen zusätzlich der belegte
Speicherplatz in GB (💾-Zeile unter den Badges, auch im Klick-Modal als
eigene Zeile) - rekursive Verzeichnisgröße, 5 Minuten gecacht (`catalog.
get_cached_size_bytes()`) und bei allen gleichzeitig bekannten Modellen
parallel statt nacheinander abgefragt, damit ein `os.walk()` über hunderte GB
nicht bei jedem WebSocket-Heartbeat den Event-Loop blockiert. Cache wird nach
jedem abgeschlossenen Download bzw. jedem "Von Platte löschen" sofort für das
betroffene Modell geleert statt bis zu 5 Minuten auf den nächsten Refresh zu
warten. Klick auf ein Modell öffnet zusätzlich ein Modal mit:
- Link zur HuggingFace-Seite (`https://huggingface.co/<model>`)
- fertigem JSON-Schnipsel im `chatLanguageModels.json`-Format (siehe VS-Code-Setup
  weiter oben) zum direkten Einfügen ins `"models"`-Array eines
  `customendpoint`-Eintrags - mit Kopieren-Button. Die `url` wird dynamisch aus der
  Adresse gebaut, über die das Dashboard gerade aufgerufen wurde.
- **"Von Platte löschen"-Button** (nur sichtbar, wenn das Modell lokal gecacht ist):
  löscht die heruntergeladenen Dateien unwiderruflich von der Platte - unabhängig
  davon, ob das Modell noch in `config.json` registriert ist oder nur lokal
  gecacht herumliegt (z.B. nach einem Entfernen im Config-Editor, siehe unten).
  Fragt vorher die tatsächliche Verzeichnisgröße ab und zeigt sie im
  Bestätigungsdialog an. Verweigert, solange das Modell noch geladen ist/lädt
  (erst entladen). Betrifft NUR die lokalen Dateien, nie die `config.json`-
  Registrierung - ein registriertes Modell bleibt danach in der Liste stehen
  (jetzt als "nicht gecacht"), bis es erneut heruntergeladen oder aus
  `config.json` entfernt wird.

### Config-Editor (`/dashboard/config`)

Erreichbar über den "⚙️ Config →"-Link oben im Haupt-Dashboard. Bearbeitet
`config.json` über echte Formularfelder (Inputs/Dropdowns/Checkboxen) statt
rohem JSON - inkl. aller Server-/Hot-Pool-/RAG-Einstellungen und einer
aufklappbaren Liste je registriertem Modell (Tool-Call-/Reasoning-Parser,
`max_model_len`, `gpu_memory_utilization`, `extra_args`, HF-Token, Notizen, ...)
mit "+ Modell hinzufügen" und Entfernen-Button pro Eintrag.

**"Remove" vs. "Von Platte löschen"** - zwei bewusst getrennte, unterschiedlich
folgenreiche Aktionen pro Modell-Eintrag:
- **Remove** entfernt nur den Eintrag aus der (noch ungespeicherten)
  Formularliste - reversibel über "Verwerfen", solange nicht gespeichert wurde.
  Die heruntergeladenen Dateien bleiben dabei unangetastet auf der Platte
  liegen (das Modell taucht danach in "Verfügbare Modelle" weiterhin als
  "lokal gecacht, nicht registriert" auf).
- **Von Platte löschen** löscht die tatsächlichen Modell-Dateien sofort und
  unwiderruflich - unabhängig von Speichern/Verwerfen hier im Formular, und
  unabhängig davon, ob der Eintrag gerade noch registriert ist oder schon
  entfernt wurde (arbeitet über den Modellnamen, nicht den Formular-Eintrag).
  Fragt vorher die Größe ab und zeigt sie im Bestätigungsdialog. Verweigert,
  solange das Modell noch geladen ist.

Für ein Modell komplett loszuwerden (Konfiguration UND Speicherplatz): erst
"Von Platte löschen" klicken, dann "Remove" + Speichern - oder umgekehrt, die
Reihenfolge spielt keine Rolle, beide Aktionen sind unabhängig voneinander.

**"🔍 Fähigkeiten automatisch erkennen"** (Button je Modell-Eintrag): liest das
lokal gecachte `chat_template.jinja` und `config.json` des Modells aus dem
HF-Cache (funktioniert auch für noch nicht registrierte, aber schon
heruntergeladene Modelle - einfach den Namen eintragen und klicken) und
schlägt Vision/Tool-Calling/Reasoning/Task samt Begründung und
Konfidenz ("wahrscheinlich"/"unsicher") vor - siehe
[`vllm_manager/capability_detector.py`](vllm_manager/capability_detector.py)
für die genaue Heuristik (u.a. `<think>`+`enable_thinking` → Qwen3-Reasoning,
`<tool_call><function=...>` → Qwen3-XML-Tool-Calling, Harmony-Channels →
gpt-oss, `1_Pooling/`-Ordner → Embedding-Task). **Reine Vermutung, nichts wird
automatisch gespeichert** - erst nach Klick auf "Vorschläge übernehmen"
landen die Werte in den Formularfeldern, gespeichert wird weiterhin nur über
den normalen Speichern-Button oben.

**Aktivierungswege** (Button oben, während des Scrollens sichtbar):
- **"Speichern (live übernehmen)"**: validiert die Eingaben serverseitig
  (Pydantic - bei Fehlern wird **nichts** geschrieben, stattdessen die
  Fehlermeldung angezeigt), sichert die bisherige `config.json` automatisch als
  Backup und übernimmt die neue Config sofort im laufenden Prozess. Wirkt für
  praktisch alles sofort (nächster Request/nächster Modellstart liest die neuen
  Werte) - **außer** `host`/`port` (der Bind-Socket des Managers selbst), die
  sind mit einem 🔁-Badge markiert und brauchen einen echten Neustart. Bereits
  laufende Engines behalten ihre alten Serve-Args, bis sie neu geladen werden.
- **"Speichern & Dienst neu starten"**: wie oben, stößt danach zusätzlich
  `sudo systemctl restart vllm` an. Braucht **passwortlosen sudo** für genau
  diesen Befehl (selbst einzurichten, z.B. per `visudo`-Eintrag - der Manager
  richtet das aus Sicherheitsgründen nicht selbst ein). Ohne passenden
  sudoers-Eintrag schlägt der Button mit einer klaren Fehlermeldung fehl, statt
  unbemerkt nichts zu tun.

**Backups** (`config_backups/` neben `config.json`, nicht Teil von Git): vor
jedem Save wird die bisherige `config.json` als `config-<Zeitstempel>.json`
gesichert (Historie der letzten 20 Änderungen, mit "Wiederherstellen"-Button
pro Zeile im Backups-Abschnitt der Seite).

**Absturzschutz beim Programmstart:** zusätzlich zur Zeitstempel-Historie wird
nach jedem erfolgreichen Start/Save eine `last_known_good.json` gepflegt. Lässt
sich `config.json` beim nächsten Start nicht mehr laden (kaputtes JSON, ungültige
Werte, von Hand editiert) - der Dienst crasht **nicht**, sondern startet
automatisch mit `last_known_good.json`, sichert die kaputte Datei zur
Fehlersuche als `config-broken-<Zeitstempel>.json` weg und zeigt eine
auffällige Warnung oben im Config-Editor an (inkl. Fehlermeldung). Die defekte
`config.json` bleibt dabei unangetastet auf der Platte liegen - ein Speichern
über den Editor (der ja die aktuell laufende, funktionierende Config anzeigt)
repariert sie beim nächsten Save automatisch wieder.

## Kostentracking (fiktiv)

Rein informativer Vergleichswert unter `/dashboard/costs` (Link "💰 Costs →" im
Haupt-Dashboard): der lokale Betrieb ist und bleibt kostenlos - diese Seite
rechnet lediglich aus, was dieselben Tokens über eine Cloud-API (Default:
offizielle **Claude-Sonnet-5**-Standardpreise, $3/$15 pro 1 Mio. Input-/Output-
Tokens) gekostet hätten. Reasoning-Tokens zählen dabei zur Ausgabe, analog zu
Anthropics eigener Abrechnung.

**Preise konfigurieren:** im [Config-Editor](#config-editor-dashboardconfig)
unter "Cost Tracking" - global (`default_pricing`) und optional pro Modell
überschreibbar (`models.<name>.pricing`, z.B. für einen Vergleich mit einem
günstigeren Referenzmodell).

**Wo die Kosten auftauchen:**
- Haupt-Dashboard, **Aktive Anfragen**: Live-Schätzung "so far" aus den bisher
  gestreamten Tokens - bewusst nur die Ausgabeseite, die Prompt-Größe ist erst
  nach Requestende bekannt.
- Haupt-Dashboard, **Letzte Anfragen**: exakte Kosten, sobald `prompt_tokens`/
  `completion_tokens` bekannt sind (siehe Caveat oben zu `stream_options.
  include_usage`) - sonst "–", bewusst keine geschätzte Zahl.
- **`/dashboard/costs`**: persistente Historie *jeder* abgeschlossenen Anfrage
  (auch fehlgeschlagene, dann ohne Kostenwert), inkl. Gesamtsumme und
  Aufschlüsselung pro Modell. Live per WebSocket. Jeder Datensatz hält den zum
  Zeitpunkt der Anfrage gültigen Preis fest (ändert sich also nicht rückwirkend,
  wenn später die Konfiguration angepasst wird - anders als die Live-Ansichten
  im Haupt-Dashboard, die immer gegen die aktuelle Preiskonfiguration rechnen).
  Datensätze einzeln löschbar, per Checkbox mehrfach auswählbar ("Alle
  auswählen" in der Kopfzeile) oder komplett zurücksetzbar.
- Persistiert als `costs.jsonl` neben `config.json` (nicht Teil von Git).

## MCP-Server (für KI-Agenten)

Erreichbar unter `http://<LAN-IP>:11434/mcp` (Streamable-HTTP-Transport). Werkzeuge:

- `list_models` – registrierte, gecachte und aktuell geladene Modelle
- `server_status` – Status aller aktiven Engines (Hot Pool)
- `pull_model(model, revision?, hf_token?)` – Download starten, gibt `job_id` zurück
- `pull_status(job_id)` – Fortschritt abfragen
- `load_model(model)` – Modell laden (blockierend)
- `unload_model(model?)` – Modell entladen (ohne Angabe: alle)
- `rag_add_text(text, collection?, source?)` / `rag_add_file(path, collection?)` /
  `rag_search(query, collection?, top_k?)` / `rag_list_collections()` /
  `rag_list_documents(collection?)` / `rag_delete_document(document_id, collection?)` /
  `rag_delete_collection(collection)` – siehe [RAG](#rag-retrieval-augmented-generation) unten

Einen MCP-Client (z.B. Claude, Hermes Agent) einfach auf diese URL zeigen lassen.

## RAG (Retrieval-Augmented Generation)

Optionale Erweiterung: eigene Texte/Dokumente in eine Wissensdatenbank ablegen und
per semantischer Suche wiederfinden (Hintergrund siehe
[docs/erklaerung-quantisierung-und-tokens-pro-sekunde.md](docs/erklaerung-quantisierung-und-tokens-pro-sekunde.md#7-was-ist-rag-und-wozu-braucht-man-es)).

**Architektur:**

```
Qdrant (Docker-Container, Port 6333/6334, --restart unless-stopped)
  ↑ speichert Vektoren + Metadaten pro Collection
vLLM-Manager (vllm_manager/rag.py)
  ↑ embedded Text über ein Embedding-Modell im Hot Pool
  ↑ (Qwen/Qwen3-Embedding-0.6B, läuft mit --runner pooling statt generate)
      ├─ /rag/* (REST, von der Dashboard-Seite genutzt)
      ├─ rag_*-MCP-Tools (für KI-Agenten)
      └─ /dashboard/rag (eigene Verwaltungsseite, EN/DE)
```

**Setup:** Qdrant läuft als Docker-Container. Erst prüfen, ob er schon existiert
(einmalig angelegte Container mit `--restart unless-stopped` überleben normale
Server-Neustarts von selbst):

```bash
docker ps -a --filter name=qdrant     # existiert er schon?

# Falls "Exited" / gestoppt:
docker start qdrant

# Nur falls er noch NIE angelegt wurde (einmalig):
docker volume create qdrant_storage
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Gesundheit prüfen:
curl http://127.0.0.1:6333/healthz     # -> "healthz check passed"
```

In `config.json` unter `"rag"`: `enabled: true`, `embedding_model` auf ein
registriertes Modell mit `"task": "embed"` gesetzt (Beispiel bereits in
`config.example.json`, im Dashboard über den Config-Editor/Abschnitt RAG
einstellbar). `chunk_size_chars`/`chunk_overlap_chars` steuern, wie Texte beim
Ablegen zerlegt werden (Default 1500/200 Zeichen, versucht an Absatz-/
Satzgrenzen zu brechen).

**Stolperstein (behoben):** Der Config-Editor hat die Dropdown-Auswahl für
`default_model` und `rag.embedding_model` bisher stillschweigend auf "kein
Modell" zurückgesetzt, sobald IRGENDEIN anderes Modell im selben Formular
bearbeitet, umbenannt oder entfernt wurde (z.B. über "🔍 Fähigkeiten
automatisch erkennen") - unabhängig davon, ob das ausgewählte Modell selbst
betroffen war. Ein anschließendes Speichern hat `rag.embedding_model` dadurch
unbemerkt auf `null` gesetzt und RAG damit faktisch deaktiviert, obwohl
`rag.enabled` weiterhin `true` blieb. Behoben: die Auswahl bleibt jetzt über
Formular-Neuaufbauten hinweg erhalten (`populateModelSelects()` in
config_dashboard.py), außer das ausgewählte Modell wird selbst entfernt.

**Nutzung:**

```bash
# Text hinzufügen
curl -X POST http://<LAN-IP>:11434/rag/collections/meine-docs/text \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "source": "notiz-vom-22.08"}'

# Datei hinzufügen (PDF/TXT/MD) - Pfad gilt auf dem Server, nicht dem Client!
curl -X POST http://<LAN-IP>:11434/rag/collections/meine-docs/file \
  -H "Content-Type: application/json" -d '{"path": "/pfad/zur/datei.pdf"}'

# Suchen
curl -X POST http://<LAN-IP>:11434/rag/collections/meine-docs/search \
  -H "Content-Type: application/json" -d '{"query": "...", "top_k": 5}'
```

Dieselben Operationen stehen auch als `rag_*`-MCP-Tools (für KI-Agenten) und über
die Dashboard-Seite **`http://<LAN-IP>:11434/dashboard/rag`** zur Verfügung
(Collections/Dokumente durchsuchen, hinzufügen, löschen, Such-Testbox).

**Wichtig:** Das Embedding-Modell wird sowohl beim Ablegen als auch bei **jeder
einzelnen Suche** neu aufgerufen (Suchanfrage muss in denselben Vektorraum
eingebettet werden) - läuft dafür ganz normal im Hot Pool mit und bleibt bei
`max_concurrent_models >= 2` neben Chat-Modellen warm.

## Von Ollama auf vLLM übernommene Modelle

Ollama wurde gestoppt und deaktiviert (`systemctl disable ollama`). Alle zuvor über
Ollama genutzten Modelle wurden auf ihr HuggingFace-Äquivalent gemappt und in
`config.json` registriert (Downloads liefen am 2026-08-22 an, Fortschritt s.o.):

| Ollama-Modell | vLLM/HuggingFace-Äquivalent | Größe | max_model_len |
|---|---|---|---|
| `qwen3:8b` | `Qwen/Qwen3-8B` | 16.4 GB | 40960 (nativ) |
| `nemotron-3-nano:4b` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` | 5.3 GB | 262144 (nativ) |
| `qwen3.8:27b` | `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 262144 (nativ) |
| `qwen3-coder:30b-a3b-q4_K_M` + `qwen3-coder-256k:latest` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 31.2 GB | 262144 (nativ) |
| `qwen3.6:35b` | `Qwen/Qwen3.6-35B-A3B-FP8` | 37.5 GB | 262144 (nativ) |
| `nemotron-3.5-lightning:latest` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 21.6 GB | 1048576 (nativ, Hybrid-Mamba) |
| `nemotron-cascade-2:latest` | `nvidia/Nemotron-Cascade-2-30B-A3B` | 63.2 GB (nur BF16 verfügbar) | 262144 (nativ) |
| `qwen3-next:80b` | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 50.8 GB | 262144 (nativ) |
| `llama3.2:latest` | `meta-llama/Llama-3.2-3B-Instruct` | 6.4 GB | **gated – noch nicht heruntergeladen, siehe unten** | 131072 (nativ) |

`max_model_len` ist jeweils auf den echten nativen Kontext des Modells gesetzt
(aus `text_config.max_position_embeddings` im jeweiligen HF `config.json`, kein
YaRN/Rope-Scaling nötig) - **nicht** künstlich auf 32768 gedeckelt wie in einer
früheren Version dieser Anleitung. Live geprüft: `Qwen3.8-27B-FP8` (262144,
1.66x KV-Cache-Concurrency-Headroom bei `gpu_memory_utilization: 0.5`) und
`Nemotron-3.5-Lightning` (1048576, 12.18x Headroom dank Hybrid-Mamba-Architektur
mit konstanter State-Größe statt klassischem Attention-KV-Cache) starten damit
sauber. Die übrigen Werte in der Tabelle sind nach demselben Muster gesetzt,
aber nicht einzeln live getestet - bei `gpu_memory_utilization`-bedingtem
Start-Fehler (KV-Cache zu klein für `max_model_len`) den Wert in `config.json`
für das jeweilige Modell senken oder `gpu_memory_utilization` erhöhen.

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
