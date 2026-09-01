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
# Zu spät gesetzt (nach den Imports) führte beim ersten Versuch dazu, dass das
# bereits lokal vorhandene 55GB-Modell komplett neu nach ~/.cache/huggingface
# heruntergeladen wurde, statt den vorhandenen Download in HF_HOME zu nutzen.
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
# quantisiert, die 48 Mamba-Layer bleiben in voller BF16-Präzision.
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

# v2: linear_attn (Mamba) NICHT mehr ignorieren - QuantizationModifier
# quantisiert diese Layer jetzt mit, aber ohne AWQ-Smoothing (weil sie nicht
# in custom_mappings vorkommen, ruft AWQ sie nie direkt auf -> kein Crash).
# Das ist quasi Round-to-Nearest für diese 48 Layer: weniger "smart" als AWQ,
# aber braucht keinen Forward-Call durch GatedDeltaNet und sollte trotzdem
# spürbar Speicherbandbreite sparen. Vision-Turm + lm_head bleiben BF16.
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
