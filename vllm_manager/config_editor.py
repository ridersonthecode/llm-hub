"""Backend für den Config-Editor im Dashboard (/dashboard/config): validiertes
Schreiben von config.json mit automatischem Backup vor jeder Änderung, plus
ein Fallback-Mechanismus für den Programmstart - wird config.json kaputt
bearbeitet (von Hand oder über einen Bug hier), landet der Dienst nicht in
einer Boot-Schleife, sondern startet mit der zuletzt bekannt guten Config und
meldet das deutlich sichtbar im Dashboard.

Zwei Backup-Arten unter config_backups/ (nicht in Git, siehe .gitignore -
enthält dieselben sensiblen Daten wie config.json selbst):
- config-<timestamp>.json: eine Momentaufnahme vor jedem Save über den Editor
  (Historie zum Zurückrollen einzelner Änderungen).
- last_known_good.json: wird nach jedem erfolgreichen Start und jedem
  erfolgreichen Save überschrieben. Das ist die Datei, auf die beim Start
  zurückgefallen wird, falls config.json nicht mehr lädt.
- config-broken-<timestamp>.json: Kopie der kaputten Datei zum Nachvollziehen,
  falls der Fallback greifen musste - config.json selbst wird dabei NICHT
  angefasst, damit ein Fix per Editor (der die aktuell laufende, funktionierende
  Config zeigt) beim nächsten Save automatisch config.json wieder repariert.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from . import capability_detector
from . import config as config_module
from .config import CONFIG_PATH, Config

logger = logging.getLogger("vllm_manager.config_editor")

# Schützt JEDEN Schreibzugriff auf config.json (save_config UND patch_config,
# siehe unten) - threading.Lock statt asyncio.Lock, weil Aufrufer teils über
# asyncio.to_thread() in einem echten Worker-Thread laufen (ein asyncio.Lock
# wirkt nur innerhalb derselben Event-Loop/desselben Threads, hier brauchen
# wir echten Cross-Thread-Ausschluss). Ohne dieses Lock konnten zwei
# gleichzeitige Schreiber (z.B. ein manueller Dashboard-Save UND eine
# automatische Hintergrund-Korrektur wie register_model_if_missing oder
# process_manager._autocorrect_kv_cache_deficit) sich gegenseitig
# überschreiben - live beobachtet: 2026-08-24, siehe patch_config()-Docstring.
_config_write_lock = threading.Lock()


class ConfigConflictError(Exception):
    """config.json wurde zwischen dem Laden (GET /config) und diesem Save-
    Versuch von anderer Stelle geändert (siehe save_config(expected_fingerprint=...)).
    Der Aufruf wird verworfen, statt die zwischenzeitliche Änderung stillschweigend
    zu überschreiben - der Nutzer soll neu laden und seine Änderung erneut eintragen."""
    pass


def config_fingerprint(cfg: Optional[Config] = None) -> str:
    """Kurzer, deterministischer Fingerabdruck des aktuell aktiven Configs (oder
    eines übergebenen), zum Erkennen zwischenzeitlicher Änderungen (siehe
    ConfigConflictError). Kein Sicherheitsmerkmal, nur Änderungserkennung -
    16 Hex-Zeichen reichen dafür locker."""
    c = cfg if cfg is not None else config_module.get_config()
    canonical = json.dumps(c.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

# Neben der tatsächlich verwendeten config.json (nicht fest PROJECT_ROOT!) -
# CONFIG_PATH kann über die Umgebungsvariable VLLM_MANAGER_CONFIG woanders
# hinzeigen (siehe config.py), die Backups sollen dem folgen.
BACKUPS_DIR = CONFIG_PATH.parent / "config_backups"
LAST_KNOWN_GOOD_PATH = BACKUPS_DIR / "last_known_good.json"
MAX_TIMESTAMPED_BACKUPS = 20

# Vom letzten Programmstart: None = config.json hat normal geladen, sonst
# eine für die UI verständliche Fehlermeldung (siehe load_config_with_fallback).
startup_warning: Optional[str] = None


def _ensure_backups_dir() -> None:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)  # atomar auf demselben Dateisystem


def _snapshot_last_known_good(cfg: Config) -> None:
    _ensure_backups_dir()
    _atomic_write_json(LAST_KNOWN_GOOD_PATH, cfg.model_dump())


def _quarantine_broken_config(error: Exception) -> Optional[str]:
    """Sichert die kaputte config.json zur Fehlersuche weg (falls vorhanden)
    und gibt den Dateinamen der Kopie zurück."""
    if not CONFIG_PATH.exists():
        return None
    _ensure_backups_dir()
    dest = BACKUPS_DIR / f"config-broken-{_timestamp()}.json"
    try:
        shutil.copy2(CONFIG_PATH, dest)
        return dest.name
    except OSError:
        logger.exception("Konnte kaputte config.json nicht sichern")
        return None


def _read_last_known_good() -> Optional[Config]:
    if not LAST_KNOWN_GOOD_PATH.exists():
        return None
    try:
        with open(LAST_KNOWN_GOOD_PATH, encoding="utf-8") as f:
            return Config(**json.load(f))
    except (OSError, json.JSONDecodeError, ValidationError):
        logger.exception("Backup last_known_good.json ist selbst kaputt")
        return None


def load_config_with_fallback() -> Config:
    """Wie config.load_config(), aber mit Absturzschutz: schlägt das Parsen/
    Validieren von config.json fehl, wird stattdessen die letzte bekannt gute
    Version geladen, statt den Dienst crashen zu lassen. Setzt bei Bedarf
    `startup_warning` für eine Banner-Anzeige im Dashboard."""
    global startup_warning
    try:
        cfg = config_module.load_config()
    except Exception as e:  # kaputtes JSON, fehlende Datei, Validierungsfehler, ...
        logger.error("config.json konnte nicht geladen werden: %s", e)
        broken_copy = _quarantine_broken_config(e)
        fallback = _read_last_known_good()
        if fallback is None:
            # Kein Backup vorhanden - können nichts retten, wie bisher crashen.
            raise
        config_module.set_config(fallback)
        startup_warning = (
            f"config.json war beim Start ungültig ({e}) - läuft aktuell mit der "
            f"zuletzt funktionierenden Backup-Version."
            + (f" Fehlerhafte Datei gesichert als config_backups/{broken_copy}." if broken_copy else "")
        )
        logger.warning(startup_warning)
        return fallback
    else:
        startup_warning = None
        _snapshot_last_known_good(cfg)
        return cfg


def _prune_backups() -> None:
    backups = sorted(BACKUPS_DIR.glob("config-*.json"), key=lambda p: p.name, reverse=True)
    # config-broken-* zählt nicht zur rollierenden Historie mit, die bleiben
    # bis sie manuell aufgeräumt werden (Diagnose-Zweck).
    timestamped = [p for p in backups if not p.name.startswith("config-broken-")]
    for old in timestamped[MAX_TIMESTAMPED_BACKUPS:]:
        try:
            old.unlink()
        except OSError:
            pass


def list_backups() -> list[dict]:
    _ensure_backups_dir()
    out = []
    for p in sorted(BACKUPS_DIR.glob("config-*.json"), key=lambda p: p.name, reverse=True):
        if p.name.startswith("config-broken-"):
            continue
        st = p.stat()
        out.append({"filename": p.name, "modified_at": st.st_mtime, "size": st.st_size})
    return out


def save_config(
    new_data: dict, *, expected_fingerprint: Optional[str] = None
) -> tuple[Config, Optional[str]]:
    """Validiert new_data, sichert die aktuelle config.json als Backup,
    schreibt new_data atomar nach config.json und übernimmt sie live ins
    laufende Programm. Wirft pydantic.ValidationError bei ungültigen Daten -
    dabei wird NICHTS geschrieben. Gibt (neue Config, Backup-Dateiname) zurück.

    expected_fingerprint (optional): Fingerabdruck (siehe config_fingerprint()),
    den der Aufrufer beim LADEN der Ausgangsdaten gesehen hat. Weicht die
    aktuell aktive Config beim Schreiben davon ab, ist config.json seither
    von anderer Stelle geändert worden (z.B. automatische Selbstkorrektur
    zwischen Laden und Klick auf "Speichern" im Dashboard-Editor) - new_data
    ist dann ein FLACHER SNAPSHOT der alten Datei und würde diese Änderung
    stillschweigend wieder rückgängig machen. Statt das zuzulassen, wird
    ConfigConflictError geworfen und NICHTS geschrieben; der Aufrufer soll
    neu laden und seine Änderung auf der frischen Version wiederholen. Wird
    aktuell vom Dashboard-Editor genutzt (siehe main.py), NICHT von
    patch_config() unten - die braucht das nicht, weil sie ohnehin immer
    frisch von der Platte liest, statt einen Snapshot zu überschreiben."""
    new_cfg = Config(**new_data)  # wirft ValidationError, wenn ungültig
    config_module.sort_models(new_cfg)  # Modelle alphabetisch, bevor geschrieben/übernommen wird

    with _config_write_lock:
        if expected_fingerprint is not None:
            current_fp = config_fingerprint()
            if current_fp != expected_fingerprint:
                raise ConfigConflictError(
                    "Die Konfiguration wurde zwischenzeitlich von anderer Stelle "
                    "geändert (z.B. eine automatische Korrektur im Hintergrund oder "
                    "ein anderer geöffneter Tab) - bitte neu laden und die eigene "
                    "Änderung erneut eintragen, um nichts zu überschreiben."
                )

        _ensure_backups_dir()
        backup_name = None
        if CONFIG_PATH.exists():
            backup_name = f"config-{_timestamp()}.json"
            shutil.copy2(CONFIG_PATH, BACKUPS_DIR / backup_name)
            _prune_backups()

        _atomic_write_json(CONFIG_PATH, new_cfg.model_dump())
        config_module.set_config(new_cfg)
        _snapshot_last_known_good(new_cfg)
        global startup_warning
        startup_warning = None  # ein erfolgreicher manueller Save entschärft die Startup-Warnung
    return new_cfg, backup_name


def patch_config(mutate: Callable[[dict], object]) -> Optional[tuple[Config, Optional[str]]]:
    """Für automatische Hintergrund-Schreiber (register_model_if_missing unten,
    process_manager._autocorrect_kv_cache_deficit) statt eines rohen
    get_config() -> model_dump() -> ... -> save_config()-Umwegs.

    Der Unterschied ist entscheidend: get_config() liefert die evtl. schon
    veraltete In-Memory-Kopie, die u.U. schon vor Sekunden (bei großen
    Modellen: nach einem kompletten Kaltstart-Versuch) geladen wurde. Baut ein
    Hintergrund-Prozess darauf einen vollständigen dict-Snapshot und schreibt
    den später komplett zurück, überschreibt er JEDE Änderung, die in der
    Zwischenzeit über den Dashboard-Editor gespeichert wurde - live
    beobachtet: 2026-08-24 (gemeldet als "ich speichere eine Änderung, aber
    nach einem Neustart ist wieder die alte Version aktiv").

    patch_config() behebt das, indem es config.json UNMITTELBAR vor dem
    Schreiben frisch von der Platte liest (nicht die zwischengespeicherte
    Kopie), `mutate(dump)` darauf anwendet und sofort - noch innerhalb
    desselben Lock-Abschnitts, also ohne dass dazwischen ein anderer Schreiber
    reinfunken kann - zurückschreibt. `mutate` darf `dump` in-place ändern;
    gibt sie explizit False zurück, wird NICHTS geschrieben (Konvention für
    "beim frischen Nachsehen war die Änderung gar nicht mehr nötig", z.B. weil
    das Modell inzwischen schon anderweitig registriert wurde) - dann liefert
    diese Funktion None statt (Config, Backup-Dateiname).

    Kein expected_fingerprint nötig (anders als save_config oben): dieser
    Pfad liest ja ohnehin garantiert frisch, es gibt also nichts, wovon er
    "veraltet" sein könnte."""
    with _config_write_lock:
        current = config_module.load_config()  # frisch von Disk, NICHT get_config()
        dump = current.model_dump()
        if mutate(dump) is False:
            return None
        new_cfg = Config(**dump)
        config_module.sort_models(new_cfg)

        _ensure_backups_dir()
        backup_name = None
        if CONFIG_PATH.exists():
            backup_name = f"config-{_timestamp()}.json"
            shutil.copy2(CONFIG_PATH, BACKUPS_DIR / backup_name)
            _prune_backups()

        _atomic_write_json(CONFIG_PATH, new_cfg.model_dump())
        config_module.set_config(new_cfg)
        _snapshot_last_known_good(new_cfg)
        global startup_warning
        startup_warning = None
    return new_cfg, backup_name


async def register_model_if_missing(model: str, note: str) -> bool:
    """Trägt `model` automatisch in config.json ein, falls dort noch nicht
    registriert - siehe Aufrufer (process_manager.ensure_loaded fürs erste
    erfolgreiche Laden, downloader._run_job für einen fertigen Download).

    Ohne das war ein Modell zwar über einen direkten Request oder das MCP-
    Tool pull_model() nutzbar/ladbar (ensure_loaded() verlangt keine
    Registrierung, vLLM lädt unbekannte Modelle einfach selbst von HF
    herunter) und tauchte in GET /models korrekt als "gecacht, aber nicht in
    config.json registriert" auf - aber eben NIE in der eigentlichen Modell-
    Konfiguration, die der Nutzer im Config-Editor sieht und pflegt (Live
    beobachtet: 2026-08-24).

    Nutzt dieselbe Fähigkeiten-Erkennung wie der "Auto-detect capabilities"-
    Button im Config-Editor (siehe capability_detector.py), damit der
    Eintrag nicht mit reinen Rate-Defaults dasteht - bestmöglicher
    Ausgangspunkt, ausdrücklich keine Garantie für Korrektheit (derselbe
    Vorbehalt wie beim manuellen Button). Best-effort: schlägt die Erkennung
    oder das Schreiben fehl, wird nur geloggt, NIE propagiert - eine
    automatische Registrierung darf den eigentlichen Lade-/Download-Vorgang
    nicht stören. Gibt True zurück, wenn tatsächlich neu registriert wurde."""
    if model in config_module.get_config().models:
        return False  # häufigster Fall - gar nicht erst blockierend nachschauen

    # Fähigkeiten-Erkennung VOR dem eigentlichen Schreiben (kann dauern - liest
    # Dateien, ggf. Netzwerk) - bewusst außerhalb von patch_config()/dessen Lock,
    # damit ein langsamer detect_capabilities() nicht alle anderen Config-
    # Schreiber blockiert. Der eigentliche Write in _mutate() unten prüft
    # `model in dump["models"]` dann nochmal gegen den zu diesem Zeitpunkt
    # frisch von der Platte gelesenen Stand (siehe patch_config()-Docstring) -
    # das hier ist also nur eine Vorab-Optimierung, keine Korrektheitsannahme.
    current = config_module.get_config()
    try:
        caps = await asyncio.to_thread(capability_detector.detect_capabilities, model, current.hf_home)
    except Exception:
        logger.exception(
            "Fähigkeiten-Erkennung für automatisch zu registrierendes Modell '%s' "
            "fehlgeschlagen - trage trotzdem mit Standardwerten ein", model,
        )
        caps = None

    entry: dict = {"notes": note}
    if caps and caps.get("found"):
        entry["task"] = caps["task"]["suggested"]
        entry["vision"] = bool(caps["vision"]["detected"])
        if caps["tool_calling"]["detected"]:
            entry["enable_auto_tool_choice"] = True
            entry["tool_call_parser"] = caps["tool_calling"]["suggested_parser"]
        if caps["reasoning"]["detected"]:
            entry["reasoning_parser"] = caps["reasoning"]["suggested_parser"]
        if caps["max_model_len"]["suggested"] is not None:
            entry["max_model_len"] = caps["max_model_len"]["suggested"]
        if caps["gpu_memory_utilization"]["suggested"] is not None:
            entry["gpu_memory_utilization"] = caps["gpu_memory_utilization"]["suggested"]

    def _mutate(dump: dict):
        if model in dump["models"]:
            return False  # zwischenzeitlich schon anderweitig registriert (z.B. paralleler Download)
        dump["models"][model] = entry

    try:
        result = await asyncio.to_thread(patch_config, _mutate)
    except Exception:
        logger.exception("Konnte '%s' nicht automatisch in config.json eintragen", model)
        return False
    registered = result is not None
    if registered:
        logger.info("Modell '%s' automatisch in config.json eingetragen (%s)", model, note)
    return registered


async def sync_detected_capabilities(model: str, caps: dict) -> bool:
    """Spiegelt task/vision/tool_calling/reasoning_parser für `model` in
    config.json - aufgerufen von process_manager._build_command() bei JEDEM
    Engine-Start mit dem dort bereits berechneten Erkennungsergebnis (kein
    zweiter Aufruf von capability_detector nötig). Anders als register_model_
    if_missing oben geht es hier NICHT ums erstmalige Anlegen, sondern ums
    stetige Aktuellhalten: diese vier Felder gelten seit 2026-08-25 als reine
    Fakten des Modells (siehe ModelConfig-Docstring), keine Nutzer-Einstellung
    mehr - der eigentliche Engine-Start nutzt `caps` direkt und liest NICHT
    mehr aus config.json, dieser Schreibvorgang dient ausschließlich der
    Dashboard-Anzeige und dem RAG-Embedding-Filter (task=="embed"). Best-
    effort, blockiert/verzögert den eigentlichen Start nie (fire-and-forget-
    Aufruf) und wird nie propagiert. Gibt True zurück, wenn tatsächlich
    geschrieben wurde (False bei bereits identischem Stand - kein unnötiger
    Backup-Churn bei jedem einzelnen Start)."""
    if not caps.get("found"):
        return False

    new_vals = {
        "task": caps["task"]["suggested"],
        "vision": bool(caps["vision"]["detected"]),
        "enable_auto_tool_choice": bool(
            caps["tool_calling"]["detected"] and caps["tool_calling"]["suggested_parser"]
        ),
        "tool_call_parser": (
            caps["tool_calling"]["suggested_parser"] if caps["tool_calling"]["detected"] else None
        ),
        "reasoning_parser": (
            caps["reasoning"]["suggested_parser"] if caps["reasoning"]["detected"] else None
        ),
    }

    def _mutate(dump: dict):
        entry = dump["models"].setdefault(model, {})
        if all(entry.get(k) == v for k, v in new_vals.items()):
            return False  # unverändert - nichts zu tun
        entry.update(new_vals)

    try:
        result = await asyncio.to_thread(patch_config, _mutate)
    except Exception:
        logger.exception("Konnte erkannte Fähigkeiten für '%s' nicht in config.json spiegeln", model)
        return False
    return result is not None


def restore_backup(filename: str) -> tuple[Config, Optional[str]]:
    """Lädt eine Datei aus config_backups/ und aktiviert sie wie save_config()
    (inkl. eigenem Backup der aktuell aktiven Config davor)."""
    _ensure_backups_dir()
    path = BACKUPS_DIR / filename
    # Nur Dateinamen ohne Pfadanteile zulassen - kein Zugriff außerhalb von
    # config_backups/ über z.B. "../config.json" o.ä.
    if path.parent != BACKUPS_DIR or not path.is_file():
        raise FileNotFoundError(filename)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return save_config(data)


def restart_service(unit: str = "vllm") -> tuple[bool, str]:
    """Stößt 'sudo systemctl restart <unit>' an (fire-and-forget, detached -
    der eigene Prozess wird dabei selbst beendet). Erfordert passwortlosen
    sudo für genau diesen Befehl (vom Nutzer selbst einzurichten, siehe
    Anleitung.md); das hier NICHT automatisch konfiguriert, da das eine
    sicherheitsrelevante System-Änderung wäre. Schlägt der Aufruf sofort fehl
    (z.B. fehlende sudo-Rechte), wird das synchron erkannt und gemeldet;
    startet er, kann der Erfolg selbst nicht mehr bestätigt werden, weil der
    eigene Prozess kurz danach stirbt."""
    try:
        proc = subprocess.Popen(
            ["sudo", "-n", "systemctl", "restart", unit],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return False, f"Befehl nicht gefunden: {e}"

    try:
        out, err = proc.communicate(timeout=1.5)
    except subprocess.TimeoutExpired:
        # Läuft noch nach 1.5s -> vermutlich unterwegs, der Restart killt uns gleich.
        return True, "Neustart ausgelöst - Dienst antwortet kurz nicht."

    if proc.returncode != 0:
        msg = (err or out or b"").decode(errors="replace").strip()
        return False, (
            f"'sudo systemctl restart {unit}' fehlgeschlagen (Exit {proc.returncode}): {msg or 'unbekannter Fehler'}. "
            f"Vermutlich fehlt passwortloser sudo für diesen Befehl - alternativ manuell ausführen: "
            f"sudo systemctl restart {unit}"
        )
    return True, "Neustart ausgelöst."
