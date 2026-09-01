# LLM-Hub auf einem neuen DGX Spark installieren

Schritt-für-Schritt-Anleitung, um dieses Projekt (inkl. RAG/Qdrant) auf einem
**zweiten** DGX Spark (NVIDIA GB10, Unified Memory) frisch aufzusetzen. Einfach
von oben nach unten durchgehen. Für Hintergrund/Details zu einzelnen Features
siehe [Anleitung.md](Anleitung.md) - dieses Dokument ist bewusst nur die reine
Installations-Checkliste.

**Was im ZIP enthalten ist:** der komplette Code (`llm_hub/`) als reiner
Datei-Snapshot (**kein** `.git` - auf dem neuen Spark ist also keine
Git-Anmeldung nötig, einfach entpacken und loslegen), Doku,
`config.example.json` (mit den heute [2026-08-24] leergetesteten/getunten
Startwerten) und die systemd-Unit-Vorlage.

**Was NICHT enthalten ist (bewusst, siehe `.gitignore`):**
- `.venv/` - wird in Schritt 3 neu gebaut (Python-Pakete sind u.a.
  architekturspezifische CUDA-Wheels, kein 1:1-Kopieren zwischen Maschinen)
- `models/` (HuggingFace-Cache, kann hunderte GB groß werden) - Modelle
  müssen auf dem neuen Spark neu heruntergeladen werden (Schritt 7)
- `models-quantized/` (eigene AWQ-INT4-Quantisierungen, ~43GB) - siehe
  [docs/anleitung-eigene-awq-quantisierung.md](docs/anleitung-eigene-awq-quantisierung.md)
  zum Nachbauen, oder einfach manuell auf den neuen Spark kopieren, falls
  vorhanden (z.B. per `rsync`/USB). Die zugehörigen Modell-Einträge in
  `config.json` sind seit 2026-08-31 "./"-relativ zu diesem Projektordner
  (z.B. `./models-quantized/Qwen3-8B-NVFP4` statt absolutem Pfad) - einfach
  die komplette `config.json` mitkopieren, kein manuelles Anpassen mehr
  nötig, solange `models-quantized/` an derselben Stelle relativ zum
  Projektordner landet (Normalfall bei einem einfachen Kopieren des ganzen
  Ordners). Ältere, noch absolute Einträge heilen sich beim ersten Start
  automatisch selbst auf die neue Form um.
- `logs/`, `config.json` (die aktuell laufende, lokale Config dieser
  Maschine), `costs.jsonl`, `last_active_model.json`

---

## 1. Voraussetzungen prüfen

Auf dem **neuen** Spark (nicht hier):

```bash
# NVIDIA-Treiber vorhanden? (sollte bei DGX-OS/Ubuntu 24.04 aarch64 schon drauf sein)
nvidia-smi
# Erwartet: Driver Version >= 580.x, CUDA Version >= 13.0
# Hinweis: memory.total/used/free zeigen hier bewusst "N/A" - das ist auf
# GB10-Unified-Memory-Systemen normal, siehe Anleitung.md.

# Python-Version
python3 --version
# Erwartet: 3.12.x (Ubuntu-24.04-Standard)

# Docker (für Qdrant/RAG, Schritt 6)
docker --version
# Falls nicht vorhanden: https://docs.docker.com/engine/install/ubuntu/
# Danach ohne sudo nutzbar machen:
#   sudo usermod -aG docker $USER && newgrp docker

# Freier Plattenplatz (Modelle sind groß - je nach geplanter Auswahl
# realistisch 50-300+ GB einplanen)
df -h ~
```

## 2. Archiv übertragen und entpacken

```bash
# Auf dieser Maschine schon erledigt: llm-hub.zip liegt bereit.
# Auf den neuen Spark kopieren, z.B. per scp:
scp llm-hub.zip <user>@<neuer-spark>:~/

# Auf dem neuen Spark:
cd ~
unzip llm-hub.zip -d llm-hub
cd llm-hub
ls    # sollte u.a. llm_hub/, INSTALL.md, config.example.json, llm-hub.service zeigen
```

## 3. Python-Umgebung aufsetzen

```bash
cd ~/llm-hub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install vllm fastapi "uvicorn[standard]" httpx huggingface_hub "mcp[cli]<2" qdrant-client pypdf
```

Dauert einige Minuten (vLLM zieht u.a. torch + CUDA-Runtime-Wheels,
zusammen mehrere GB). Zur Referenz, diese Versionen laufen auf der
Quell-Maschine stabil: `vllm==0.28.0`, `torch==2.13.0` (Stand 2026-09-01).

**Smoke-Test** (bricht früh ab, falls die GPU nicht sauber erkannt wird,
statt erst bei einem echten Modell-Ladeversuch):

```bash
python3 -c "
import torch
free, total = torch.cuda.mem_get_info()
print(f'CUDA ok - {free/1024**3:.1f}/{total/1024**3:.1f} GiB frei')
"
```

## 4. Konfiguration anlegen

```bash
cd ~/llm-hub
cp config.example.json config.json
```

In `config.json` mindestens anpassen:

| Feld | Wert |
|---|---|
| `default_model` | ein Modell, das du tatsächlich herunterladen willst (Default-Vorschlag: `NousResearch/Hermes-3-Llama-3.1-8B`, klein und bekannt funktionierend) |

`hf_home` NICHT anpassen (bzw. auf `null` lassen) - `null` bedeutet
automatisch "`<Projektordner>/models`", was auf dem neuen Spark schon von
allein an der richtigen Stelle liegt, egal wohin du das Projekt entpackst.
Nur bei explizitem Wunsch nach einem HF-Cache außerhalb des Projektordners
(z.B. andere/größere Platte) hier einen absoluten Pfad setzen.

Alles andere (Hot-Pool-Größe, GPU-Speicherbudget, Modell-Liste, RAG) ist
bereits mit den heute getesteten/gefixten Werten vorbelegt - Details siehe
[Anleitung.md](Anleitung.md#configjson). Nicht benötigte Modelle einfach auf
`"enabled": false` lassen oder aus der `"models"`-Liste entfernen.

Optional, falls gated Modelle (z.B. Llama) genutzt werden sollen: siehe
[Anleitung.md](Anleitung.md#llama-32-nachträglich-aktivieren).

## 5. Manueller Probe-Start (ohne systemd)

```bash
cd ~/llm-hub
source .venv/bin/activate
python -m llm_hub
```

Erwartet: Der Prozess startet sofort (Modelle laden erst bei Bedarf), Logs
laufen im Terminal. In einem zweiten Terminal prüfen:

```bash
curl http://127.0.0.1:11434/health
curl http://127.0.0.1:11434/models
```

Mit `Strg+C` wieder beenden, sobald das klappt - weiter mit Schritt 6+7 für
Dauerbetrieb.

## 6. RAG: Qdrant als Docker-Container

Nur nötig, falls RAG genutzt werden soll (`"rag": {"enabled": true}` in
`config.json`, ist im mitgelieferten `config.example.json` bereits so
vorbelegt):

```bash
docker volume create qdrant_storage
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Gesundheit prüfen:
curl http://127.0.0.1:6333/healthz
# -> "healthz check passed"
```

`--restart unless-stopped` sorgt dafür, dass der Container jeden
Server-Neustart automatisch übersteht - kein weiterer manueller Schritt
nötig. Ohne RAG-Bedarf: `"rag": {"enabled": false}` in `config.json` setzen
und diesen Schritt überspringen.

## 7. Als systemd-Dienst einrichten (Dauerbetrieb)

```bash
cd ~/llm-hub
cp llm-hub.service /tmp/llm-hub.service
# User/Group/WorkingDirectory/ExecStart-Pfad auf den eigenen Benutzernamen anpassen:
sed -i "s#mwagner#$USER#g; s#/home/$USER/vllm#$HOME/vllm#g" /tmp/llm-hub.service
cat /tmp/llm-hub.service   # kurz gegenprüfen, ob Pfade stimmen

sudo mv /tmp/llm-hub.service /etc/systemd/system/llm-hub.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm

# Status prüfen:
sudo systemctl status llm-hub
sudo journalctl -u vllm -f   # Live-Logs, Strg+C zum Beenden
```

**Optional, aber empfohlen** - passwortloser `sudo` für den
"Speichern & Dienst neu starten"-Button im Config-Editor
(`/dashboard/config`, siehe Anleitung.md):

```bash
sudo visudo -f /etc/sudoers.d/vllm-restart
# Zeile einfügen (Benutzername anpassen):
#   <user> ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart llm-hub
```

## 8. Verifizieren

```bash
# Von einer anderen Maschine im Netz (LAN-IP des neuen Spark einsetzen):
curl http://<neuer-spark>:11434/health

# Dashboard im Browser öffnen:
#   http://<neuer-spark>:11434/dashboard
```

Im Dashboard sollte "Loaded Models" leer sein (noch kein Modell
heruntergeladen) und "Available Models" die Liste aus `config.json` zeigen,
jeweils mit Badge "not cached".

## 9. Erstes Modell laden

```bash
# Download anstoßen (Modellname aus config.json, z.B. das Default-Modell):
curl -X POST http://127.0.0.1:11434/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "NousResearch/Hermes-3-Llama-3.1-8B"}'
# -> {"job_id": "..."}

# Fortschritt im Dashboard verfolgen, oder:
curl http://127.0.0.1:11434/models/pull/<job_id>

# Nach Abschluss testen (lädt das Modell automatisch, dauert beim ersten
# Mal 1-5 Minuten Kaltstart):
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "NousResearch/Hermes-3-Llama-3.1-8B",
       "messages": [{"role": "user", "content": "Hallo, wer bist du?"}]}'
```

Weitere Modelle: genauso per `POST /models/pull`, oder komfortabler über
den Config-Editor (`/dashboard/config`) + "Laden"-Button im Dashboard.

## 10. Optional: VS-Code-MCP-Integration

Falls dieses Projekt auch als MCP-Server für VS Code Copilot Chat genutzt
werden soll (siehe [Anleitung.md](Anleitung.md#mcp-server-für-ki-agenten)):
in `.vscode/mcp.json` die URL auf die LAN-IP des **neuen** Spark anpassen
(steht aktuell noch auf der IP der alten Maschine):

```json
{
  "servers": {
    "llm-hub": {
      "type": "http",
      "url": "http://<neuer-spark>:11434/mcp"
    }
  }
}
```

## 11. Sicherheit (kurz)

Standardmäßig: kein API-Key, `0.0.0.0:11434` - erreichbar für jeden mit
Netzwerkroute zu dieser Maschine, keine Firewall-Regel aktiv. Details und
wie man das einschränkt: [Anleitung.md, Abschnitt "Sicherheit"](Anleitung.md#sicherheit).

---

**Fertig.** Ab hier gilt die normale [Anleitung.md](Anleitung.md) für den
Alltagsbetrieb (Modelle verwalten, Dashboard, Config-Editor, RAG nutzen,
Wiederholungsschleifen-Schutz, Kostentracking, ...).
