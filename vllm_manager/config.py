"""Lädt und hält die zentrale config.json (kein sudo nötig zum Ändern)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("VLLM_MANAGER_CONFIG", PROJECT_ROOT / "config.json"))


class ApiKeyConfig(BaseModel):
    enabled: bool = False
    key: str = ""


class Pricing(BaseModel):
    """Rein fiktive Kostenkalkulation zum Vergleich mit einer Cloud-API (siehe
    cost_tracker.py) - hat KEINERLEI Einfluss auf den tatsächlichen (kostenlosen)
    lokalen Betrieb. Default: offizielle Claude-Sonnet-5-Standardpreise
    (USD pro 1 Mio. Tokens, Stand siehe Anthropic-Preisliste)."""
    input_per_mtok: float = 3.0
    output_per_mtok: float = 15.0  # gilt auch für Reasoning-Tokens, wie bei Anthropics eigener Abrechnung


class ModelConfig(BaseModel):
    tool_call_parser: Optional[str] = None
    # Trennt vLLMs Antwort in "reasoning_content" (Denkprozess) und "content"
    # (eigentliche Antwort), analog zu DeepSeek-R1s API. Ohne das landet bei
    # Thinking-Modellen (z.B. Qwen3) der komplette <think>...</think>-Block
    # als normaler Text im Chat, statt in der einklappbaren Reasoning-Box.
    # Gültige Werte: vllm serve --help=all | grep -A2 reasoning-parser, z.B.
    # "qwen3", "deepseek_r1", "granite", "mistral", ...
    reasoning_parser: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_auto_tool_choice: bool = False
    vision: bool = False
    # "generate" (normales Chat-/Completion-Modell) oder "embed"
    # (Embedding-Modell für RAG, z.B. Qwen3-Embedding). Steuert, ob vllm serve
    # mit --runner pooling gestartet wird (vLLMs Nachfolger von --task).
    task: str = "generate"
    extra_args: list[str] = Field(default_factory=list)
    hf_token: Optional[str] = None
    notes: Optional[str] = None
    enabled: bool = True
    # Sicherheitsnetz gegen durchgehende Generierungen (siehe main.py:
    # _apply_default_max_tokens - ein Modell in einer Wiederholungsschleife
    # hat einmal 80.000+ Tokens am Stück produziert, 43 Minuten lang, weil
    # weder Client noch Server eine Obergrenze gesetzt hatten). Greift NUR,
    # wenn der Request selbst kein max_tokens/max_completion_tokens mitschickt
    # - ein expliziter Client-Wunsch wird nie heruntergedrückt. None = kein
    # Limit (wie bisher, nur durch max_model_len begrenzt).
    max_tokens: Optional[int] = None
    # Override für dieses Modell - None = default_pricing (siehe Config unten) gilt.
    pricing: Optional[Pricing] = None


class RagConfig(BaseModel):
    """Konfiguration für die optionale RAG-Erweiterung (Qdrant + Embedding-Modell)."""

    enabled: bool = False
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    # Muss ein in "models" registriertes Modell mit task:"embed" sein.
    embedding_model: Optional[str] = None
    default_collection: str = "default"
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 200


class Config(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11434
    engine_host: str = "127.0.0.1"
    engine_port: int = 18811
    hf_home: str = str(PROJECT_ROOT / "models")
    vllm_bin: Optional[str] = None
    api_key: ApiKeyConfig = Field(default_factory=ApiKeyConfig)
    idle_timeout_seconds: Optional[int] = None
    # Anzahl gleichzeitig laufender vLLM-Engine-Prozesse ("Hot Pool"). Bei 1
    # (Default) verhält sich der Manager wie zuvor: exklusiv, jeder Wechsel ist
    # ein Kaltstart. Bei >1 belegt jede Engine einen eigenen Port
    # (engine_port, engine_port+1, ...) und bereits geladene Modelle bleiben
    # beim Wechsel warm - solange gpu_memory_ceiling nicht überschritten wird.
    max_concurrent_models: int = 1
    # Obergrenze für die Summe der gpu_memory_utilization aller gleichzeitig
    # laufenden Engines (gemeinsame Unified Memory auf der GB10 - jede Engine
    # reserviert ihren Anteil unabhängig, ohne von den anderen zu wissen).
    # Wird diese beim Laden eines neuen Modells überschritten, wird die am
    # längsten ungenutzte Engine zuerst verdrängt (LRU).
    gpu_memory_ceiling: float = 0.90
    # Ollama-artiges Verhalten: das zuletzt genutzte Modell wird beim nächsten
    # Dienststart automatisch im Hintergrund nachgeladen (siehe process_manager.py
    # load_last_active_model()/_persist_last_active()). Auf false setzen, um nach
    # einem Neustart bewusst mit leerem Hot Pool zu starten.
    auto_reload_last_model: bool = True
    default_model: Optional[str] = None
    startup_timeout_seconds: int = 900
    default_serve_args: dict = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    rag: RagConfig = Field(default_factory=RagConfig)
    # Fallback-Preise fürs fiktive Kostentracking (siehe cost_tracker.py) für
    # Modelle ohne eigenen models.<name>.pricing-Override.
    default_pricing: Pricing = Field(default_factory=Pricing)

    def resolved_vllm_bin(self) -> str:
        if self.vllm_bin:
            return self.vllm_bin
        return str(Path(sys.executable).parent / "vllm")

    def serve_args_for(self, model: str) -> tuple[float, int]:
        mcfg = self.models.get(model)
        defaults = self.default_serve_args or {}
        gmu = (mcfg.gpu_memory_utilization if mcfg else None)
        if gmu is None:
            gmu = defaults.get("gpu_memory_utilization", 0.5)
        mml = (mcfg.max_model_len if mcfg else None)
        if mml is None:
            mml = defaults.get("max_model_len", 32768)
        return float(gmu), int(mml)


_config: Optional[Config] = None


def load_config(path: Path = CONFIG_PATH) -> Config:
    global _config
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _config = Config(**data)
    return _config


def set_config(cfg: Config) -> Config:
    """Übernimmt eine bereits validierte Config direkt ins Live-Objekt, ohne
    erneut von der Platte zu lesen (genutzt vom Config-Editor nach dem
    Schreiben, siehe config_editor.py - vermeidet ein unnötiges zweites
    Parsen/Validieren derselben Daten)."""
    global _config
    _config = cfg
    return _config


def get_config() -> Config:
    if _config is None:
        return load_config()
    return _config
