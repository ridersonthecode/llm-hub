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


class ModelConfig(BaseModel):
    tool_call_parser: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_auto_tool_choice: bool = False
    vision: bool = False
    extra_args: list[str] = Field(default_factory=list)
    hf_token: Optional[str] = None
    notes: Optional[str] = None
    enabled: bool = True


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
    default_model: Optional[str] = None
    startup_timeout_seconds: int = 900
    default_serve_args: dict = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

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


def get_config() -> Config:
    if _config is None:
        return load_config()
    return _config
