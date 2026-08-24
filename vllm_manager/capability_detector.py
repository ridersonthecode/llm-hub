"""Best-effort Erkennung von Modell-Fähigkeiten (Vision/Tool-Calling/Reasoning/
Task/max_model_len/gpu_memory_utilization) aus lokal gecachten HuggingFace-
Metadaten (chat_template.jinja bzw. das darin eingebettete Template in
tokenizer_config.json, sowie config.json/model.safetensors.index.json) - reine
Heuristik als AUSGANGSPUNKT, KEIN Ersatz für eine manuelle Prüfung. Diese
Funktion selbst schreibt nichts - zwei Aufrufer nutzen sie:
- Config-Editor, Button "Fähigkeiten automatisch erkennen": zeigt die
  Vorschläge samt Begründung im Dashboard, der Nutzer übernimmt und
  speichert sie explizit.
- config_editor.register_model_if_missing(): trägt ein neu heruntergeladenes/
  erstmals geladenes, noch unregistriertes Modell automatisch mit diesen
  Vorschlägen in config.json ein (siehe dort) - hier OHNE manuelle
  Bestätigung, deshalb sind die notes des Eintrags entsprechend markiert.

Die Erkennung ist zwangsläufig unvollständig: es gibt dutzende Chat-Template-
Dialekte (siehe die vielen reasoning_parser/tool_call_parser-Namen in vLLM
selbst). Hier werden nur die verbreitetsten, eindeutig erkennbaren Muster
abgedeckt (Qwen3-Familie inkl. Nemotron-3.x-Reuse, Harmony/gpt-oss, Mistral,
Llama-3-Pythonic) - alles andere landet als "erkannt, aber Format unklar" statt
einer falschen Vermutung. max_model_len/gpu_memory_utilization sind reine
technische Schätzungen (siehe _detect_max_model_len/_detect_gpu_memory_
utilization) ohne jede Kenntnis vom tatsächlichen Nutzungsmuster."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def _model_dir(model: str, hf_home: str) -> Optional[Path]:
    """Neuester Snapshot-Ordner für ein Modell im HF-Cache, oder None falls
    nicht gecacht. Lokale Pfade (z.B. selbst quantisierte Modelle außerhalb
    des HF-Cache, siehe models-quantized/) werden direkt als Verzeichnis
    behandelt."""
    p = Path(model)
    if p.is_absolute() and p.is_dir():
        return p
    cache_dir = Path(hf_home) / "hub" / ("models--" + model.replace("/", "--"))
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [d for d in snapshots.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)  # neueste Revision bei mehreren


def _read_chat_template(model_dir: Path) -> str:
    direct = model_dir / "chat_template.jinja"
    if direct.exists():
        try:
            return direct.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    tok_cfg = model_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        try:
            data = json.loads(tok_cfg.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            return ""
        tpl = data.get("chat_template", "")
        if isinstance(tpl, list):
            for entry in tpl:
                if isinstance(entry, dict) and entry.get("name") == "default":
                    return entry.get("template", "")
            if tpl and isinstance(tpl[0], dict):
                return tpl[0].get("template", "")
            return ""
        return tpl or ""
    return ""


def _read_hf_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return {}


# Verschachtelte Sub-Configs, in denen Multi-Modal-Architekturen (Vision+Text)
# ihre eigentlichen Text-Modell-Parameter (max_position_embeddings etc.)
# ablegen, statt sie auf Top-Level zu haben - live beobachtet bei Qwen3.8-27B
# (model_type "qwen3_5", vision_config + text_config statt flacher Struktur).
# Reihenfolge = Suchreihenfolge, erster Treffer gewinnt.
_NESTED_TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config")


def _detect_max_model_len(hf_cfg: dict) -> dict:
    """max_position_embeddings ist die vom Modell ARCHITEKTONISCH unterstützte
    Obergrenze, keine Empfehlung für einen bestimmten Anwendungsfall - ein
    sehr hoher Wert (manche Modelle >1M) erhöht den KV-Cache-Speicherbedarf
    entsprechend. Wird trotzdem 1:1 als Vorschlag übernommen (das ist bereits
    das bestehende Muster bei den manuell gepflegten Modellen in config.json:
    max_model_len = max_position_embeddings), aber mit deutlichem Hinweis in
    der evidence, damit im Config-Editor klar ist, dass das kein Sicherheits-
    Check gegen den tatsächlich verfügbaren Speicher ist."""
    val = hf_cfg.get("max_position_embeddings")
    source = "config.json"
    if val is None:
        for key in _NESTED_TEXT_CONFIG_KEYS:
            sub = hf_cfg.get(key)
            if isinstance(sub, dict) and sub.get("max_position_embeddings") is not None:
                val = sub["max_position_embeddings"]
                source = f"config.json.{key}"
                break
    if not isinstance(val, int) or val <= 0:
        return {"suggested": None, "confidence": "unknown", "evidence": None}
    return {
        "suggested": val,
        "confidence": "high",
        "evidence": (
            f"max_position_embeddings={val} in {source} gefunden - architektonische Obergrenze "
            f"des Modells, keine Speicher-Prüfung. Bei sehr großen Werten steigt der KV-Cache-"
            f"Speicherbedarf entsprechend; bei OOM hier reduzieren."
        ),
    }


def _model_weights_bytes(model_dir: Path) -> Optional[int]:
    """Gesamtgröße der Modell-Gewichte aus dem safetensors-Index (steht dort
    bereits vor-berechnet als metadata.total_size, kein Aufsummieren aller
    Shard-Dateien nötig). Fällt auf das Aufsummieren der *.safetensors-
    Dateigrößen im Verzeichnis zurück, falls kein Index existiert (Modelle
    mit nur einer einzigen .safetensors-Datei haben oft keinen)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8", errors="ignore"))
            total = idx.get("metadata", {}).get("total_size")
            if isinstance(total, (int, float)) and total > 0:
                return int(total)
        except (OSError, json.JSONDecodeError):
            pass
    try:
        total = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
        return total or None
    except OSError:
        return None


def _total_device_memory_bytes() -> Optional[int]:
    """Gesamter GPU-Speicher in Bytes - siehe process_manager._query_gpu_
    memory_gib() für dieselbe torch.cuda.mem_get_info()-Quelle. Bewusst NICHT
    nvidia-smi (liefert auf diesem Unified-Memory-System kein memory.total).
    None, falls torch/CUDA hier (API-Server-Prozess, kein Engine-Prozess)
    aus irgendeinem Grund nicht verfügbar ist - dann bleibt gpu_memory_
    utilization unbestimmt statt einer möglicherweise falschen Schätzung."""
    try:
        import torch
        _free, total = torch.cuda.mem_get_info()
        return int(total)
    except Exception:
        return None


def _detect_gpu_memory_utilization(model_dir: Path) -> dict:
    """Schätzt eine KONSERVATIVE UNTERGRENZE für gpu_memory_utilization aus
    der reinen Gewichtsgröße + einem pauschalen Aufschlag für Aktivierungen/
    CUDA-Graph/minimalen KV-Cache (live gemessen bei einem 35B-Modell: ca.
    3.2 GiB für Aktivierung+CUDAGraph zusätzlich zu den reinen Gewichten,
    hier grob mit +20% pauschal abgedeckt). Das ist bewusst ein Minimalwert
    zum sicheren Starten mit wenig Kontext/Parallelität, KEINE Empfehlung
    für produktiven Durchsatz - große Kontextfenster oder mehrere
    gleichzeitige Anfragen brauchen mehr KV-Cache-Speicher, also einen
    höheren Wert als hier vorgeschlagen."""
    weights = _model_weights_bytes(model_dir)
    total = _total_device_memory_bytes()
    if not weights or not total:
        return {"suggested": None, "confidence": "unknown", "evidence": None}
    needed = weights * 1.2  # Gewichte + Aktivierungs-/CUDA-Graph-/Mindest-KV-Cache-Puffer
    fraction = needed / total
    # Auf 2 Nachkommastellen aufgerundet (nie knapper als geschätzt), min. 0.05.
    suggested = max(0.05, math.ceil(fraction * 100) / 100)
    if suggested >= 1.0:
        return {
            "suggested": None, "confidence": "low",
            "evidence": (
                f"Geschätzter Speicherbedarf ({needed / 1024**3:.1f} GiB) übersteigt den gesamten "
                f"GPU-Speicher ({total / 1024**3:.1f} GiB) - keine sinnvolle Schätzung möglich, "
                f"manuell prüfen (Quantisierung? anderes System?)."
            ),
        }
    return {
        "suggested": round(suggested, 2),
        "confidence": "low",
        "evidence": (
            f"Konservativer Minimalwert aus Gewichtsgröße ({weights / 1024**3:.1f} GiB) + 20% Puffer "
            f"für Aktivierungen/CUDA-Graph/minimalen KV-Cache, bezogen auf {total / 1024**3:.1f} GiB "
            f"Gesamt-GPU-Speicher dieses Systems. Reicht zum sicheren Start mit wenig Kontext/"
            f"Parallelität - für mehr Kontextfenster oder mehrere gleichzeitige Anfragen erhöhen."
        ),
    }


def _detect_reasoning(template: str) -> dict:
    if not template:
        return {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None}
    if "<|channel|>" in template and "analysis" in template:
        return {
            "detected": True, "suggested_parser": "openai_gptoss", "confidence": "high",
            "evidence": "Harmony-Format (<|channel|>analysis) im chat_template.jinja gefunden.",
        }
    if "<think>" in template:
        if "enable_thinking" in template:
            return {
                "detected": True, "suggested_parser": "qwen3", "confidence": "high",
                "evidence": "<think>-Tag + enable_thinking-Umschalter gefunden (Qwen3-Format, u.a. auch von "
                             "neueren Nemotron-Modellen wiederverwendet).",
            }
        return {
            "detected": True, "suggested_parser": None, "confidence": "low",
            "evidence": "<think>-Tag gefunden, aber Format nicht sicher zuordenbar - manuell prüfen.",
        }
    return {
        "detected": False, "suggested_parser": None, "confidence": "high",
        "evidence": "Kein <think>-Tag im chat_template.jinja gefunden.",
    }


def _detect_tool_calling(template: str) -> dict:
    if not template:
        return {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None}
    if "<tool_call>" in template and "<function=" in template:
        return {
            "detected": True, "suggested_parser": "qwen3_xml", "confidence": "high",
            "evidence": "<tool_call><function=...>-Format gefunden (Qwen3/Nemotron-verschachteltes XML).",
        }
    if "[TOOL_CALLS]" in template:
        return {
            "detected": True, "suggested_parser": "mistral", "confidence": "high",
            "evidence": "[TOOL_CALLS]-Marker gefunden (Mistral-Format).",
        }
    if "<|python_tag|>" in template:
        return {
            "detected": True, "suggested_parser": "llama3_json", "confidence": "high",
            "evidence": "<|python_tag|>-Marker gefunden (Llama-3-Format).",
        }
    if "<tool_call>" in template:
        return {
            "detected": True, "suggested_parser": "hermes", "confidence": "low",
            "evidence": "<tool_call>-Tag gefunden, generisches JSON-Format vermutet (Hermes als verbreiteter "
                         "Default) - Format ggf. abweichend, prüfen.",
        }
    if "tools" in template and "function" in template.lower():
        return {
            "detected": True, "suggested_parser": None, "confidence": "low",
            "evidence": "Tool-Unterstützung im Template erkennbar, aber Format unklar - manuell prüfen.",
        }
    return {
        "detected": False, "suggested_parser": None, "confidence": "high",
        "evidence": "Keine Tool-Call-Syntax im chat_template.jinja gefunden.",
    }


_VISION_ARCH_HINTS = ("vl", "vision", "omni", "image")


def _detect_vision(hf_cfg: dict, model: str) -> dict:
    architectures = hf_cfg.get("architectures") or []
    arch_str = " ".join(architectures).lower()
    has_vision_keys = any(k in hf_cfg for k in ("vision_config", "vision_encoder", "vision_tower", "mm_vision_tower"))
    if has_vision_keys or any(h in arch_str for h in _VISION_ARCH_HINTS):
        return {
            "detected": True, "confidence": "high",
            "evidence": "vision_config bzw. eine Vision-Architektur in config.json gefunden.",
        }
    if any(h in model.lower() for h in ("vl", "-vision", "vision-", "omni")):
        return {
            "detected": True, "confidence": "low",
            "evidence": "Modellname deutet auf Vision hin, aber config.json bestätigt es nicht - unsicher.",
        }
    return {
        "detected": False, "confidence": "high" if hf_cfg else "unknown",
        "evidence": "Keine Vision-Architektur in config.json gefunden." if hf_cfg else None,
    }


def _detect_task(hf_cfg: dict, model: str, model_dir: Path) -> dict:
    architectures = hf_cfg.get("architectures") or []
    is_causal = any(("ForCausalLM" in a or "ForConditionalGeneration" in a) for a in architectures)
    has_pooling_dir = (model_dir / "1_Pooling").exists()  # sentence-transformers-Konvention
    name_hint = "embed" in model.lower()
    if has_pooling_dir:
        return {
            "suggested": "embed", "confidence": "high",
            "evidence": "sentence-transformers-Pooling-Layout (1_Pooling/) im Cache gefunden.",
        }
    if is_causal:
        return {
            "suggested": "generate", "confidence": "high",
            "evidence": "ForCausalLM/ForConditionalGeneration-Architektur in config.json gefunden.",
        }
    if name_hint:
        return {
            "suggested": "embed", "confidence": "low",
            "evidence": "Modellname deutet auf Embedding hin, aber nicht über Architektur bestätigt - unsicher.",
        }
    return {
        "suggested": "generate", "confidence": "low" if hf_cfg else "unknown",
        "evidence": "Keine eindeutigen Hinweise gefunden, \"generate\" als Standardannahme.",
    }


def detect_capabilities(model: str, hf_home: str) -> dict:
    """Gibt für vision/tool_calling/reasoning/task/max_model_len/gpu_memory_
    utilization jeweils detected-oder-suggested + confidence ("high"/"low"/
    "unknown") + evidence (Begründung als Text) zurück. `found: false`, wenn
    das Modell lokal gar nicht gecacht ist - dann gibt es nichts, woraus sich
    etwas ableiten ließe."""
    model_dir = _model_dir(model, hf_home)
    if model_dir is None:
        return {
            "found": False,
            "vision": {"detected": False, "confidence": "unknown", "evidence": None},
            "tool_calling": {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None},
            "reasoning": {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None},
            "task": {"suggested": "generate", "confidence": "unknown", "evidence": None},
            "max_model_len": {"suggested": None, "confidence": "unknown", "evidence": None},
            "gpu_memory_utilization": {"suggested": None, "confidence": "unknown", "evidence": None},
        }

    template = _read_chat_template(model_dir)
    hf_cfg = _read_hf_config(model_dir)
    task = _detect_task(hf_cfg, model, model_dir)

    tool_calling = _detect_tool_calling(template)
    reasoning = _detect_reasoning(template)
    if task["suggested"] == "embed" and task["confidence"] == "high":
        # Embedding-Modelle erben oft ein generisches Chat-Template vom
        # Basismodell, das nie tatsächlich in diesem Modus genutzt wird (siehe
        # z.B. Qwen3-Embedding) - Tool-Calling/Reasoning wären hier reines
        # Rauschen aus dem geerbten Template, nicht eine echte Fähigkeit.
        tool_calling = {
            "detected": False, "suggested_parser": None, "confidence": "high",
            "evidence": "Embedding-Modell (task=embed) - Tool-Calling/Reasoning nicht anwendbar, "
                         "unabhängig vom (ggf. vom Basismodell geerbten) chat_template.jinja.",
        }
        reasoning = dict(tool_calling)

    return {
        "found": True,
        "vision": _detect_vision(hf_cfg, model),
        "tool_calling": tool_calling,
        "reasoning": reasoning,
        "task": task,
        "max_model_len": _detect_max_model_len(hf_cfg),
        "gpu_memory_utilization": _detect_gpu_memory_utilization(model_dir),
    }
