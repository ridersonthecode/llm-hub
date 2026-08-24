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
    # Automatisches server-seitiges RAG (siehe rag.apply_auto_rag): falls
    # gesetzt (Name einer existierenden Collection, siehe /dashboard/rag),
    # wird bei JEDER Chat-Anfrage an dieses Modell automatisch in dieser
    # Collection gesucht (letzte User-Nachricht als Suchtext) und relevante
    # Treffer als Kontext vorangestellt - der Client (VS Code, curl, ...)
    # bekommt davon nichts mit und muss nichts Besonderes unterstützen. None =
    # kein automatisches RAG für dieses Modell (Default). Live änderbar über
    # den Config-Editor, kein Neustart nötig. Setzt rag.enabled +
    # rag.embedding_model voraus (siehe RagConfig unten), sonst wird still
    # übersprungen.
    rag_collection: Optional[str] = None
    # Vorbeugung gegen Wiederholungsschleifen im Denkprozess/Antwort (siehe
    # main.py: _apply_default_repetition_penalty). vLLM-eigener Sampling-
    # Parameter (kein OpenAI-Standardfeld, aber von vLLMs Server direkt
    # unterstützt) - Werte >1.0 bestrafen bereits erzeugte Tokens, senken also
    # die Wahrscheinlichkeit exakter Wiederholungen. Greift NUR, wenn der
    # Request selbst kein repetition_penalty mitschickt. None = kein
    # server-seitiger Default (wie bisher, reines vLLM-/Client-Verhalten).
    # Sinnvoller Startwert bei anfälligen Modellen: 1.1-1.3.
    repetition_penalty: Optional[float] = None
    # Sicherheitsnetz gegen Wiederholungsschleifen: liest den Antwort-Stream
    # live mit (main.py gen()) und bricht die Anfrage ab, sobald derselbe
    # Text-Abschnitt mehrfach hintereinander exakt wiederkehrt - typisches
    # Symptom kleinerer Reasoning-Modelle, die im Denkprozess (<think>...)
    # hängen bleiben. Nur für gestreamte Anfragen wirksam (siehe main.py-
    # Docstring). Default an; pro Modell abschaltbar, falls es zu
    # Fehlalarmen bei absichtlich repetitiven Antworten kommt.
    repetition_detection: bool = True
    # Priorität bei der Verdrängungsauswahl in process_manager._make_room()
    # (Hot Pool voll oder Speicherbudget knapp): HÖHERE Zahl = wird SPÄTER
    # verdrängt (bevorzugt schlafen gelegt statt beendet, oder ganz
    # verschont). Bei gleicher Priorität entscheidet weiterhin LRU (am
    # längsten ungenutztes Modell zuerst) wie bisher. Default 0 = keine
    # explizite Präferenz, verhält sich wie vor Einführung dieses Felds.
    # Wird im Config-Editor per Drag-and-Drop-Liste gesetzt, nicht von Hand
    # gepflegt - siehe cfg.section.priority.
    priority: int = 0


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
    # Schwellenwerte fürs automatische server-seitige RAG (siehe
    # ModelConfig.rag_collection oben / rag.apply_auto_rag) - gelten global
    # für alle Modelle mit gesetzter rag_collection. auto_rag_min_score
    # verhindert, dass bei einer inhaltlich unpassenden Frage sinnlos
    # (irrelevanter) Kontext eingefügt wird - kostet Tokens und kann sogar von
    # der eigentlichen Antwort ablenken.
    auto_rag_top_k: int = 3
    auto_rag_min_score: float = 0.5
    # Collection für die automatische "Lessons Learned"-Ablage (siehe
    # mcp_tools.remember_lesson/search_lessons): KI-Agenten mit Tool-Zugriff
    # (z.B. Qwen in VS Code Copilot Chat, Agent-Modus, via .vscode/mcp.json)
    # speichern hier gelöste Fehler/Korrekturen, damit sie beim nächsten
    # ähnlichen Problem sofort wiedergefunden werden - projektweit, nicht nur
    # in der jeweiligen Chat-Session.
    lessons_learned_collection: str = "lessons_learned"
    # Sicherheitsnetz zusätzlich zum aktiven Tool-Aufruf: wird bei JEDER
    # Chat-Anfrage an ein Modell mit gesetzter ModelConfig.rag_collection
    # automatisch AUCH die Lessons-Learned-Collection durchsucht und
    # eingespeist (siehe rag.apply_auto_rag) - greift auch dann, wenn das
    # Modell search_lessons nicht selbst aufruft.
    lessons_learned_auto_inject: bool = True


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
    startup_timeout_seconds: int = 1800
    # vLLMs eingebauter Sleep Mode (siehe process_manager.py sleep_engine/
    # wake_engine): bei Verdrängung wird eine Engine bevorzugt schlafen gelegt
    # statt beendet - der Prozess bleibt am Leben (kein Kaltstart, kein
    # Re-JIT der Kernel, kein erneuter CUDA-Graph-Capture beim nächsten
    # Aufwecken nötig), nur die Gewichte werden aus dem GPU-Speicher entfernt.
    # Live getestet auf NVIDIA GB10 (Unified Memory): Aufwecken eines 35B-
    # Modells dauerte ~2s statt ~290s Kaltstart. Setzt voraus, dass die
    # installierte vLLM-Version --enable-sleep-mode unterstützt (ab ca. 0.7) -
    # bei Problemen (z.B. "Sleep mode is not supported on current platform")
    # hier auf false setzen, dann verhält sich der Pool wie zuvor (nur
    # hartes Verdrängen).
    enable_sleep_mode: bool = True
    # Wie lange eine Anfrage für ein NOCH NICHT geladenes Modell in der
    # Warteschlange wartet, falls gerade kein Platz im Hot Pool ist (alle
    # anderen Engines sind entweder selbst im Kaltstart oder bearbeiten
    # gerade eine Anfrage und werden deshalb nicht verdrängt) - siehe
    # process_manager._make_room(). Statt die Anfrage sofort mit einem
    # Fehler abzulehnen, wird gewartet, bis eine Engine frei wird. Anfragen
    # an ein bereits geladenes/ladendes Modell warten NIE in dieser
    # Warteschlange, nur echte Modellwechsel.
    queue_timeout_seconds: int = 1800
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

    def priority_for(self, model: str) -> int:
        """Siehe ModelConfig.priority - 0 (neutral) für unbekannte/nicht
        registrierte Modelle, z.B. per HF-Repo-Name direkt geladen ohne
        eigenen models.<name>-Eintrag."""
        mcfg = self.models.get(model)
        return mcfg.priority if mcfg else 0

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


def sort_models(cfg: Config) -> Config:
    """Sortiert cfg.models alphabetisch (case-insensitive) nach Modellname, in
    place. Dict-Reihenfolge in Python bestimmt die Reihenfolge überall dort,
    wo einfach über cfg.models.items() iteriert wird (Dashboard, main.py,
    mcp_tools.py, ...) sowie die Reihenfolge beim Serialisieren nach
    config.json - ein zentraler Sortierpunkt reicht daher aus, statt an jeder
    Anzeigestelle einzeln zu sortieren."""
    cfg.models = dict(sorted(cfg.models.items(), key=lambda kv: kv[0].casefold()))
    return cfg


def load_config(path: Path = CONFIG_PATH) -> Config:
    global _config
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _config = sort_models(Config(**data))
    return _config


def set_config(cfg: Config) -> Config:
    """Übernimmt eine bereits validierte Config direkt ins Live-Objekt, ohne
    erneut von der Platte zu lesen (genutzt vom Config-Editor nach dem
    Schreiben, siehe config_editor.py - vermeidet ein unnötiges zweites
    Parsen/Validieren derselben Daten)."""
    global _config
    _config = sort_models(cfg)
    return _config


def get_config() -> Config:
    if _config is None:
        return load_config()
    return _config
