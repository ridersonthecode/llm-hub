"""Lädt und hält die zentrale config.json (kein sudo nötig zum Ändern)."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("LLM_HUB_CONFIG", PROJECT_ROOT / "config.json"))


# --- Portabilität: lokale Modellpfade relativ zu PROJECT_ROOT statt absolut -
# Eigene lokale Modelle (z.B. selbst quantisierte AWQ/NVFP4-Varianten unter
# models-quantized/, siehe catalog.py/nvfp4_quantizer.py) werden in
# config.json als Dict-KEY unter "models" geführt - dort IST der Modellname
# der Pfad (siehe catalog.cache_dir_for). Ein absoluter Pfad wie
# "/home/user/llm-hub/models-quantized/Foo" macht die ganze config.json
# maschinenspezifisch: kopiert man sie unverändert auf eine andere Maschine
# oder installiert an einem anderen Ort, zeigt der Eintrag ins Leere (siehe
# INSTALL.md, dort bisher als bewusst manueller Schritt dokumentiert).
#
# Fix: "./"-Präfix bedeutet "relativ zu PROJECT_ROOT" - übersteht jedes Kopieren
# des Projektordners an einen beliebigen Ort, ohne dass irgendetwas in
# config.json angepasst werden müsste (Chat vom 2026-08-31: "es muss nachher
# so sein, dass alle Pfade exakt zum neuen System passen, egal wohin ich es
# installiere"). Ein echter absoluter Pfad ("/...") bleibt zusätzlich
# unterstützt, für den seltenen Fall eines Modells bewusst außerhalb des
# Projektordners (z.B. andere/größere Platte) - der ist dann weiterhin
# maschinenspezifisch, das lässt sich ohne Wissen über das Zielsystem nicht
# auflösen.
def resolve_local_model_path(model: str) -> Optional[Path]:
    """Löst einen "model"-String zu einem lokalen Dateisystempfad auf, falls
    es sich um ein lokales Modell handelt (./-relativ oder absolut) - None,
    falls es ein HuggingFace-Hub-Repo-Name ("org/name") ist."""
    if model.startswith("./") or model.startswith("../"):
        return (PROJECT_ROOT / model).resolve()
    if model.startswith("/"):
        return Path(model)
    return None


def local_model_key_for(path: Path) -> str:
    """Kehrt resolve_local_model_path() um: baut aus einem absoluten Pfad den
    portablen "model"-Key fürs Registrieren in config.json (z.B. nach einer
    NVFP4-Quantisierung, siehe nvfp4_quantizer.py). "./"-relativ, falls der
    Pfad innerhalb PROJECT_ROOT liegt (der Normalfall - models-quantized/
    liegt direkt im Projektordner), sonst absoluter Fallback."""
    path = path.resolve()
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return str(path)
    return "./" + rel.as_posix()


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
    # tool_call_parser, reasoning_parser, enable_auto_tool_choice, vision und
    # task (weiter unten) sind seit 2026-08-25 KEINE Nutzer-Einstellungen mehr,
    # auch wenn sie hier weiter als Felder existieren: es sind Fakten der
    # Modell-Architektur selbst (aus chat_template.jinja/config.json erkannt,
    # siehe capability_detector.py), keine Abwägung wie z.B. gpu_memory_
    # utilization. process_manager._build_command() erkennt sie bei JEDEM
    # Engine-Start frisch und ignoriert dabei komplett, was hier steht -
    # dieselbe Heuristik, die früher der "Auto-detect"-Button im Config-Editor
    # anzeigte, läuft jetzt automatisch bei jedem Start. Die Werte hier werden
    # nur noch als Kopie fürs Dashboard/RAG-Filter (task=="embed") mitgeführt
    # (siehe config_editor.sync_detected_capabilities) - im Config-Editor
    # deshalb nicht mehr editierbar, nur noch als "erkannt"-Anzeige zu sehen.
    # Nutzer-Feedback dazu: "es ergibt keinen Sinn, Werte anzupassen, die
    # feststehen."
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_auto_tool_choice: bool = False
    vision: bool = False
    # Startet die Engine mit --enforce-eager (deaktiviert CUDA-Graph-Capture
    # beim Kaltstart, siehe process_manager._build_command). Verkürzt den
    # Kaltstart spürbar (live gemessen: ~35s von ~290s bei einem 35B-Modell
    # waren reine Graph-Capture/Kernel-Kompilierung), macht die Engine
    # dauerhaft aber langsamer (jeder Forward-Pass läuft über den vollen
    # PyTorch-Eager-Dispatch statt eines vorkompilierten CUDA-Graphs) - echter
    # Trade-off Kaltstart-Geschwindigkeit vs. Dauer-Durchsatz, kein Freebie.
    # Lohnt sich vor allem für selten/sporadisch genutzte Modelle, die oft
    # KOMPLETT kaltstarten (Slot-Verdrängung, volles Idle-Timeout-Entladen).
    # Default aus.
    fast_load: bool = False
    # Begrenzt, für welche Batch-Größen vLLM beim Kaltstart CUDA-Graphen
    # aufzeichnet (siehe process_manager._build_command, Flag
    # --cudagraph-capture-sizes). Ohne dieses Feld leitet vLLM die Liste aus
    # max_num_seqs (Default 256) ab: [1,2,4] + 8er-Schritte bis 256 + 16er-
    # Schritte bis 512 - satte ~51 Größen, für die JEWEILS ein echter
    # Forward-Pass durchs Modell läuft, nur um den Graphen aufzuzeichnen.
    # Live gemessen: davon stammten ~35s der ~290s Kaltstart eines 35B-Modells.
    # Für einen Heimserver, der selten mehr als eine Handvoll gleichzeitiger
    # Sequenzen bedient, ist das massive Overkill. Kommagetrennte Liste
    # realistischer Batch-Größen (z.B. "1,2,4,8,16") reduziert die Capture-
    # Zeit etwa proportional zur Anzahl der Größen, OHNE die Dauer-
    # Geschwindigkeit im abgedeckten Bereich zu verschlechtern - anders als
    # fast_load also kein Kompromiss im Normalbetrieb, nur bei seltenen
    # Lastspitzen jenseits der größten angegebenen Zahl fällt vLLM für die
    # überzähligen Sequenzen auf den langsameren Eager-Pfad zurück. Wird
    # ignoriert, wenn fast_load aktiv ist (dort gibt es ohnehin keine
    # CUDA-Graphen). None/leer = vLLMs Standardliste (kein Override).
    cudagraph_capture_sizes: Optional[str] = None
    # "generate" (normales Chat-/Completion-Modell) oder "embed"
    # (Embedding-Modell für RAG, z.B. Qwen3-Embedding). Steuert, ob vllm serve
    # mit --runner pooling gestartet wird (vLLMs Nachfolger von --task). Wie
    # oben beschrieben: kein Nutzer-Wert mehr, wird bei jedem Start neu erkannt.
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
    # Docstring). Vor dem endgültigen Abbruch startet main.py automatisch bis
    # zu _LOOP_RETRY_MAX frische Neuversuche im selben Stream (höherer
    # repetition_penalty + neuer seed je Versuch) - für den Client sichtbar
    # nur als Zwischen-Hinweis, kein Fehler/Timeout. Default an; pro Modell
    # abschaltbar, falls es zu Fehlalarmen bei absichtlich repetitiven
    # Antworten kommt.
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
    # Welche Inference-Engine process_manager._build_command() für dieses
    # Modell startet. "vllm" (Default) = bisheriges Verhalten (`vllm serve`).
    # "sglang" = `<sglang_python> -m sglang.launch_server ...` (siehe
    # Config.sglang_python) statt vLLM - für Modelle/Checkpoints, die (noch)
    # keinen vLLM-Support haben, aber ein OpenAI-kompatibles API wie vLLM
    # bereitstellen (z.B. NVFP4-Checkpoints für DGX Spark, die einen
    # gepatchten SGLang-Fork brauchen). Bewusst schmal gehalten: Auto-detect
    # Capabilities/Perf-Tuning (tool_call_parser, reasoning_parser,
    # cudagraph_capture_sizes) sind vLLM-Flag-Namen und werden für "sglang"
    # NICHT an die Engine übergeben (siehe process_manager._build_sglang_
    # command) - bei Bedarf über extra_args selbst ergänzen (z.B.
    # "--tool-call-parser", "qwen25").
    engine: Literal["vllm", "sglang"] = "vllm"


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


class Config(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11434
    engine_host: str = "127.0.0.1"
    engine_port: int = 18811
    # None = automatisch PROJECT_ROOT/"models" (siehe resolved_hf_home()) -
    # bewusst kein fest eingebrannter absoluter Pfad wie früher: Config.
    # model_dump() (siehe config_editor.save_config) serialisiert bei JEDEM
    # Speichern den kompletten Zustand nach config.json - ein String-Default
    # würde also schon beim ersten Speichern überhaupt egal welchen anderen
    # Feldes als absoluter Pfad DIESER Maschine eingefroren, selbst wenn nie
    # bewusst gesetzt (Chat vom 2026-08-31, Portabilitäts-Analyse). Gleiches
    # Muster wie vllm_bin/sglang_python unten - nur explizit gesetzt bedeutet
    # "woanders als im Projektordner".
    hf_home: Optional[str] = None
    vllm_bin: Optional[str] = None
    # Pfad zum Python-Interpreter, der für Modelle mit ModelConfig.engine ==
    # "sglang" `-m sglang.launch_server` ausführt (siehe process_manager.
    # _build_sglang_command). Meist NICHT dasselbe venv wie der Manager
    # selbst/vllm_bin - SGLang (erst recht ein gepatchter Fork für ein
    # bestimmtes Checkpoint-Format) bringt eigene, teils inkompatible
    # Abhängigkeiten mit und lebt üblicherweise in einem eigenen venv. None =
    # Fallback auf sys.executable (nur sinnvoll, falls SGLang tatsächlich im
    # selben venv wie der Manager installiert ist).
    sglang_python: Optional[str] = None
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
    # Wie lange eine Anfrage für ein NOCH NICHT geladenes Modell in der
    # Warteschlange wartet, falls gerade kein Platz im Hot Pool ist (alle
    # anderen Engines sind entweder selbst im Kaltstart oder bearbeiten
    # gerade eine Anfrage und werden deshalb nicht verdrängt) - siehe
    # process_manager._make_room(). Statt die Anfrage sofort mit einem
    # Fehler abzulehnen, wird gewartet, bis eine Engine frei wird. Anfragen
    # an ein bereits geladenes/ladendes Modell warten NIE in dieser
    # Warteschlange, nur echte Modellwechsel.
    queue_timeout_seconds: int = 1800
    # Obergrenze gleichzeitig BEARBEITETER Anfragen, über ALLE Modelle hinweg -
    # unabhängig vom Hot Pool (max_concurrent_models begrenzt nur, wie viele
    # Modelle gleichzeitig GELADEN sind, nicht wie viele Anfragen parallel
    # laufen). Trifft eine (max_concurrent_requests+1)-te Anfrage ein, wartet
    # sie in einer Warteschlange (siehe request_queue.py), statt sofort mit
    # allen anderen um GPU-Zeit zu konkurrieren. Default 2: auf einem
    # Heimserver mit einer GPU/Unified-Memory-System bringt echte Parallelität
    # jenseits weniger Anfragen ohnehin kaum noch etwas (geteilte Rechenzeit,
    # geteilter Speicher), macht aber jede einzelne Anfrage langsamer.
    max_concurrent_requests: int = 2
    # Siehe max_concurrent_requests oben. Eine wartende Anfrage wird NICHT
    # sofort übergeben, sobald ein Slot frei wird, sondern erst, nachdem für
    # mindestens so viele Sekunden GAR KEINE neue Anfrage mehr eingetroffen
    # ist (über alle Anfragen hinweg, nicht nur die wartende selbst) - siehe
    # request_queue.py-Moduldocstring für den Grund (Clients wie VS Code/
    # GitHub Copilot Chat schicken bei Retry/Abbruch/Tool-Aufrufen oft mehrere
    # Anfragen in schneller Folge; ohne diese Ruhephase würde der Manager
    # womöglich genau die Anfrage an ein Modell übergeben, die der Client
    # Sekundenbruchteile später selbst schon wieder verwirft). 0 = kein
    # Debounce, wartende Anfragen werden sofort bei freiem Slot übergeben.
    queue_debounce_seconds: float = 3.0
    default_serve_args: dict = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    rag: RagConfig = Field(default_factory=RagConfig)
    # Fallback-Preise fürs fiktive Kostentracking (siehe cost_tracker.py) für
    # Modelle ohne eigenen models.<name>.pricing-Override.
    default_pricing: Pricing = Field(default_factory=Pricing)

    def resolved_hf_home(self) -> str:
        if self.hf_home:
            return self.hf_home
        return str(PROJECT_ROOT / "models")

    def resolved_vllm_bin(self) -> str:
        if self.vllm_bin:
            return self.vllm_bin
        return str(Path(sys.executable).parent / "vllm")

    def resolved_sglang_python(self) -> str:
        if self.sglang_python:
            return self.sglang_python
        return sys.executable

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


def _migrate_local_model_keys(cfg: Config) -> bool:
    """Schreibt Modell-Keys, die als absoluter Pfad INNERHALB PROJECT_ROOT
    hinterlegt sind, auf die portable "./"-Form um (siehe local_model_key_for/
    resolve_local_model_path oben) - heilt eine bestehende config.json aus der
    Zeit vor dieser Umstellung automatisch mit, ohne manuellen Eingriff. Gibt
    True zurück, falls sich dabei etwas geändert hat (load_config()
    persistiert dann einmalig zurück auf die Platte)."""
    changed = False
    renamed: dict[str, "ModelConfig"] = {}
    for key, mcfg in cfg.models.items():
        if key.startswith("/") and Path(key).is_relative_to(PROJECT_ROOT):
            new_key = local_model_key_for(Path(key))
            if new_key != key and new_key not in cfg.models and new_key not in renamed:
                renamed[new_key] = mcfg
                changed = True
                continue
        renamed[key] = mcfg
    if changed:
        cfg.models = renamed
    return changed


def _migrate_hf_home(cfg: Config) -> bool:
    """Setzt hf_home zurück auf None (= automatisch), falls es exakt dem
    entspricht, was resolved_hf_home() ohnehin liefern würde - ein reiner
    No-op für das LAUFENDE Verhalten (auf DIESER Maschine identisch), aber
    entfernt den eingefrorenen absoluten Pfad aus config.json, bevor sie
    z.B. auf eine andere Maschine kopiert wird (siehe hf_home-Docstring in
    ModelConfig oben - jedes Speichern über den Config-Editor friert den
    damals aufgelösten Wert sonst dauerhaft ein)."""
    if cfg.hf_home and cfg.hf_home == str(PROJECT_ROOT / "models"):
        cfg.hf_home = None
        return True
    return False


def load_config(path: Path = CONFIG_PATH) -> Config:
    global _config
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = Config(**data)
    migrated = _migrate_local_model_keys(cfg)
    migrated = _migrate_hf_home(cfg) or migrated
    _config = sort_models(cfg)
    if migrated:
        try:
            _atomic_write_json(path, _config.model_dump())
        except OSError:
            logging.getLogger("llm_hub.config").exception(
                "Konnte migrierte lokale Modellpfade nicht nach %s zurückschreiben - "
                "wirkt trotzdem für die laufende Sitzung, nur nicht dauerhaft.", path,
            )
    return _config


def _atomic_write_json(path: Path, data: dict) -> None:
    # Gleiches Format wie config_editor._atomic_write_json (bewusst identisch
    # gehalten - beide schreiben dieselbe Datei).
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


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
