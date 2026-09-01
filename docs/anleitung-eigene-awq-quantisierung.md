# Eigene AWQ-INT4-Quantisierung erstellen (am Beispiel Qwen3.8-27B)

Diese Anleitung dokumentiert Schritt für Schritt, wie `Qwen/Qwen3.8-27B-AWQ-INT4-v2`
entstanden ist: aus den offiziellen BF16-Original-Gewichten selbst quantisiert,
weil es dafür keine vLLM-taugliche 4-bit-Version von HuggingFace gab (nur
Ollamas GGUF, das vLLM nicht mehr laden kann – siehe
[Erklärung GGUF vs. AWQ](erklaerung-quantisierung-und-tokens-pro-sekunde.md)).

Ergebnis laut [Benchmark](benchmark-ollama-vs-vllm-qwen3.8-27b.md): 74% von
Ollamas Decode-Geschwindigkeit bei fast identischer Dateigröße (18GB vs. 17.7GB).

## Voraussetzungen

- **Speicherplatz:** mindestens ~140GB frei (55.6GB Original-BF16-Download +
  ~18-25GB Ergebnis + Zwischenspeicher während der Quantisierung).
- **GPU mit genug Speicher**, um das BF16-Modell komplett zu laden (bei einem
  27B-Modell ~55GB) plus Kalibrierungs-Overhead. Auf der GB10 hier mit 121GB
  Unified Memory kein Problem.
- Das Original-Modell ist bereits lokal vorhanden (Schritt 1) oder wird
  frisch heruntergeladen.

## Schritt 1: Offizielle BF16-Original-Gewichte herunterladen

Nicht die quantisierte Version (z.B. `Qwen/Qwen3.8-27B-FP8`) verwenden, sondern
das unquantisierte Original – nur daraus kann man sinnvoll neu quantisieren.

Über den LLM-Hub (nutzt den bestehenden `HF_HOME`-Cache, mit Fortschrittsanzeige):

```bash
curl -X POST http://127.0.0.1:11434/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.8-27B"}'
# -> {"job_id": "..."}

# Fortschritt abfragen:
curl http://127.0.0.1:11434/models/pull/<job_id>
```

Alternativ ganz ohne LLM-Hub, direkt mit `huggingface_hub`:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B', cache_dir='/pfad/zu/deinem/hf_home')
"
```

Das Modell landet unter `<HF_HOME>/hub/models--Qwen--Qwen3.8-27B/snapshots/<hash>/`
(18 Safetensor-Shards, ~55.6GB gesamt).

## Schritt 2: Isolierte Python-Umgebung aufsetzen

**Wichtig:** nicht in der venv installieren, die der produktive vLLM-Dienst
nutzt! `llm-compressor` verlangt eine andere `compressed-tensors`-Version als
vLLM (Konflikt), das würde den laufenden Dienst kaputt machen.

```bash
cd /home/mwagner/llm-hub
python3 -m venv .venv-quantize
.venv-quantize/bin/pip install --upgrade pip
.venv-quantize/bin/pip install llmcompressor torch transformers accelerate torchvision
```

`torchvision` wird gebraucht, weil Qwen3.8-27B ein Vision-Language-Modell ist –
`llm-compressor` versucht beim Start automatisch einen Bild/Video-Prozessor zu
initialisieren, auch wenn man nur Text zur Kalibrierung nutzt.

## Schritt 3: Das Quantisierungs-Skript

Speichern unter z.B. `.venv-quantize/quantize_qwen38.py`. Pfade (`MODEL_ID`,
`OUTPUT_DIR`, der `HF_HOME`-Pfad, der Pfad zur `config.json` des Originalmodells)
ggf. anpassen.

```python
"""Einmaliges Quantisierungs-Skript: Qwen/Qwen3.8-27B (BF16, offiziell) -> AWQ
INT4 (vLLM-kompatibel), mit llm-compressor. Läuft in der isolierten
.venv-quantize (NICHT die produktive .venv von llm-hub.service!).

Hybrid-Architektur (Attention + Gated-Delta-Net/Mamba-Layer abwechselnd) - AWQ
quantisiert generisch pro nn.Linear-Layer, unabhängig vom umgebenden Mechanismus.
Vision-Turm (model.visual.*) und lm_head bleiben unquantisiert (Standard-Praxis,
sensibel für Qualitätsverlust, vergleichsweise klein)."""
import os

# WICHTIG: muss vor JEDEM Import von transformers/datasets/huggingface_hub
# gesetzt werden - die lesen HF_HOME beim Import und cachen den Pfad intern.
# Zu spät gesetzt (nach den Imports) führt dazu, dass das bereits lokal
# vorhandene 55GB-Modell komplett neu nach ~/.cache/huggingface heruntergeladen
# wird, statt den vorhandenen Download in HF_HOME zu nutzen.
os.environ["HF_HOME"] = "/home/mwagner/llm-hub/models"

import json
from glob import glob

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"
OUTPUT_DIR = "/home/mwagner/llm-hub/models-quantized/Qwen3.8-27B-AWQ-INT4-v2"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 2048

print("Lade Tokenizer/Processor...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

print("Lade Modell (BF16, das dauert etwas)...", flush=True)
# Qwen3.8-27B ist ein Vision-Language-Modell (architectures:
# Qwen3_5ForConditionalGeneration) - AutoModelForCausalLM würde die falsche
# (text-only) Klasse laden. AutoModelForImageTextToText matcht korrekt.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype="bfloat16", device_map="auto", trust_remote_code=True, local_files_only=True
)
print("Modell geladen. Module (Top-Level):", [n for n, _ in model.named_children()], flush=True)

print("Lade Kalibrierungs-Datensatz...", flush=True)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding=False)


ds = ds.map(preprocess, remove_columns=ds.column_names)

# Hybrid-Architektur (64 Layer: 16 "full_attention" + 48 "linear_attention" =
# Gated-Delta-Net/Mamba). llm-compressors automatische Mapping-Erkennung
# erzeugt für Mamba-Layer eine Zuordnung, deren "Smoothing"-Schritt versucht,
# das GatedDeltaNet-Modul direkt aufzurufen - dessen forward()-Signatur passt
# nicht zum generischen Aufrufschema (TypeError: missing 1 required positional
# argument 'hidden_states'). Deshalb hier eigene Mappings OHNE die
# linear_attention-Zuordnung: nur full_attention- und MLP-Layer werden per AWQ
# "gesmoothed", die Mamba-Layer laufen unten über den QuantizationModifier ohne
# Smoothing (siehe recipe weiter unten).
cfg = json.load(open(glob("/home/mwagner/llm-hub/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/config.json")[0]))
layer_types = cfg["text_config"]["layer_types"]
full_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
full_re = "|".join(str(i) for i in full_indices)
print(f"full_attention Layer (AWQ-quantisiert): {full_indices}", flush=True)
print(f"linear_attention/Mamba Layer (RTN-quantisiert, kein AWQ-Smoothing): {[i for i, t in enumerate(layer_types) if t == 'linear_attention']}", flush=True)

custom_mappings = [
    AWQMapping(
        f"re:.*layers\\.({full_re})\\.input_layernorm$",
        ["re:.*self_attn.q_proj$", "re:.*self_attn.k_proj$", "re:.*self_attn.v_proj$"],
    ),
    AWQMapping("re:.*post_attention_layernorm$", ["re:.*gate_proj$", "re:.*up_proj$"]),
    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
]

# Mamba-Layer (linear_attn) NICHT ignorieren - QuantizationModifier
# quantisiert diese Layer mit, aber ohne AWQ-Smoothing (weil sie nicht in
# custom_mappings vorkommen, ruft AWQ sie nie direkt auf -> kein Crash). Das
# ist quasi Round-to-Nearest für diese Layer: weniger "smart" als AWQ, aber
# braucht keinen Forward-Call durch GatedDeltaNet und spart trotzdem spürbar
# Speicherbandbreite. Vision-Turm + lm_head bleiben BF16.
recipe = [
    AWQModifier(mappings=custom_mappings, duo_scaling="both"),
    QuantizationModifier(
        ignore=["lm_head", "re:.*visual.*", "re:.*vision.*"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

print("Starte AWQ-Quantisierung (kann 30-90+ Minuten dauern)...", flush=True)
# WICHTIG: kein output_dir hier - oneshot()s Auto-Save unterstützt kein
# save_original_format=False, was wir wegen des device_map="auto"-Offloadings
# brauchen (sonst RuntimeError beim Speichern: "could not revert some weight
# conversions because of offloading"). Stattdessen unten manuell speichern.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print(f"Quantisierung fertig, speichere nach {OUTPUT_DIR}...", flush=True)
model.save_pretrained(OUTPUT_DIR, save_compressed=True, save_original_format=False)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fertig gespeichert.", flush=True)
```

## Schritt 4: Ausführen

Läuft lange (Kalibrierung über 128 Beispiele + Quantisierung + Speichern,
insgesamt ca. eine Stunde) – am besten im Hintergrund mit Logging:

```bash
cd /home/mwagner/llm-hub
nohup .venv-quantize/bin/python .venv-quantize/quantize_qwen38.py > quantize.log 2>&1 &
tail -f quantize.log
```

Am Ende sollte in etwa stehen:

```
Quantisierung fertig, speichere nach /home/mwagner/llm-hub/models-quantized/Qwen3.8-27B-AWQ-INT4-v2...
Fertig gespeichert.
```

## Schritt 5: Fehlende Zusatzdateien kopieren

`model.save_pretrained()` speichert nur die Gewichte + Tokenizer, aber nicht
alle Hilfsdateien (Bild-/Video-Prozessor-Konfiguration, `vocab.json`,
`merges.txt`), die für ein Vision-Language-Modell noch gebraucht werden:

```bash
SRC=$(ls -d /home/mwagner/llm-hub/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/)
OUT=/home/mwagner/llm-hub/models-quantized/Qwen3.8-27B-AWQ-INT4-v2
cp "${SRC}preprocessor_config.json" "${SRC}video_preprocessor_config.json" \
   "${SRC}vocab.json" "${SRC}merges.txt" "$OUT/"
```

Kurzer Check, dass die Quantisierungs-Metadaten korrekt geschrieben wurden:

```bash
python3 -c "
import json
d = json.load(open('$OUT/config.json'))
qc = d.get('quantization_config', {})
print('format:', qc.get('format'), 'quant_method:', qc.get('quant_method'))
"
# Erwartete Ausgabe: format: pack-quantized quant_method: compressed-tensors
```

## Schritt 6: In vLLM registrieren und testen

Im LLM-Hub: neuen Eintrag in `config.json` unter `"models"`, **Pfad als
Modell-Name** (kein HuggingFace-Repo, also lokaler Pfad) - seit 2026-08-31
"./"-relativ zum Projektordner statt absolut, damit `config.json` beim
Kopieren auf eine andere Maschine/an einen anderen Ort portabel bleibt (siehe
config.py, `resolve_local_model_path`):

```json
"./models-quantized/Qwen3.8-27B-AWQ-INT4-v2": {
  "tool_call_parser": "qwen3_xml",
  "max_model_len": 262144,
  "gpu_memory_utilization": 0.4,
  "enable_auto_tool_choice": true,
  "vision": true,
  "enabled": true,
  "notes": "Selbst quantisiert, siehe docs/anleitung-eigene-awq-quantisierung.md"
}
```

Dienst neu starten, laden, testen:

```bash
sudo systemctl restart llm-hub

curl -X POST "http://127.0.0.1:11434/models/.%2Fmodels-quantized%2FQwen3.8-27B-AWQ-INT4-v2/load"

curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "./models-quantized/Qwen3.8-27B-AWQ-INT4-v2",
  "messages": [{"role":"user","content":"Sag nur OK."}],
  "max_tokens": 10
}'
```

Ohne LLM-Hub, direkt mit `vllm serve`:

```bash
vllm serve /home/mwagner/llm-hub/models-quantized/Qwen3.8-27B-AWQ-INT4-v2 \
  --max-model-len 262144 --gpu-memory-utilization 0.4
```

## Stolpersteine (4 Anläufe waren nötig)

Falls beim Nachbauen ähnliche Fehler auftauchen:

1. **`HF_HOME` zu spät gesetzt** → Modell wird komplett neu heruntergeladen
   statt aus dem lokalen Cache geladen zu werden. Fix: `os.environ["HF_HOME"]`
   ganz an den Anfang des Skripts, vor alle Imports.
2. **`ImportError: ... Torchvision library ...`** → `torchvision` fehlt in der
   venv (wird für die automatische Prozessor-Initialisierung bei
   Vision-Language-Modellen gebraucht, auch bei reiner Text-Kalibrierung).
   Fix: `pip install torchvision` (gleiche CUDA-Version wie das installierte
   `torch` beachten).
3. **`TypeError: Qwen3_5GatedDeltaNet.forward() missing 1 required positional
   argument: 'hidden_states'`** → AWQs automatische Mapping-Erkennung für
   Hybrid-Attention-Modelle versucht, das Mamba-Modul direkt aufzurufen; dessen
   `forward()`-Signatur passt nicht zum generischen Aufrufschema. Fix: eigene
   `AWQMapping`-Liste ohne die `linear_attention`-Zuordnung übergeben (siehe
   Skript oben, `custom_mappings`).
4. **`RuntimeError: We could not revert some weight conversions because of
   offloading...`** beim Speichern → tritt auf, weil `device_map="auto"` das
   Modell über mehrere Shards/Geräte verteilt. Fix: `oneshot()` **ohne**
   `output_dir` aufrufen (kein Auto-Save), stattdessen manuell
   `model.save_pretrained(OUTPUT_DIR, save_compressed=True,
   save_original_format=False)`.

## Ergebnis einordnen

Nicht alle 64 Layer sind gleich gut quantisiert: die 16 `full_attention`-Layer
und alle MLP-Blöcke laufen über echtes AWQ (mit "Smoothing", siehe
[Erklärung](erklaerung-quantisierung-und-tokens-pro-sekunde.md)), die 48
Mamba-Layer nur über einfaches Round-to-Nearest (kein Smoothing). Das erklärt,
warum das Ergebnis schnell, aber nicht ganz auf Ollama-Niveau ist – siehe
[Benchmark](benchmark-ollama-vs-vllm-qwen3.8-27b.md) für die genauen Zahlen.
