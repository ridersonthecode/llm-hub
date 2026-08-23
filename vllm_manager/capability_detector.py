"""Best-effort Erkennung von Modell-Fähigkeiten (Vision/Tool-Calling/Reasoning/
Task) aus lokal gecachten HuggingFace-Metadaten (chat_template.jinja bzw. das
darin eingebettete Template in tokenizer_config.json, sowie config.json) -
reine Heuristik als AUSGANGSPUNKT für den Config-Editor (Button "Fähigkeiten
automatisch erkennen"), KEIN Ersatz für eine manuelle Prüfung. Nichts wird
automatisch in config.json geschrieben - der Nutzer sieht die Vorschläge samt
Begründung im Dashboard und muss sie explizit übernehmen und speichern.

Die Erkennung ist zwangsläufig unvollständig: es gibt dutzende Chat-Template-
Dialekte (siehe die vielen reasoning_parser/tool_call_parser-Namen in vLLM
selbst). Hier werden nur die verbreitetsten, eindeutig erkennbaren Muster
abgedeckt (Qwen3-Familie inkl. Nemotron-3.x-Reuse, Harmony/gpt-oss, Mistral,
Llama-3-Pythonic) - alles andere landet als "erkannt, aber Format unklar" statt
einer falschen Vermutung."""
from __future__ import annotations

import json
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
    """Gibt für vision/tool_calling/reasoning/task jeweils detected/suggested +
    confidence ("high"/"low"/"unknown") + evidence (Begründung als Text)
    zurück. `found: false`, wenn das Modell lokal gar nicht gecacht ist -
    dann gibt es nichts, woraus sich etwas ableiten ließe."""
    model_dir = _model_dir(model, hf_home)
    if model_dir is None:
        return {
            "found": False,
            "vision": {"detected": False, "confidence": "unknown", "evidence": None},
            "tool_calling": {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None},
            "reasoning": {"detected": False, "suggested_parser": None, "confidence": "unknown", "evidence": None},
            "task": {"suggested": "generate", "confidence": "unknown", "evidence": None},
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
    }
