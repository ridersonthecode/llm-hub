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

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from . import config as config_module
from .config import CONFIG_PATH, Config

logger = logging.getLogger("vllm_manager.config_editor")

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


def save_config(new_data: dict) -> tuple[Config, Optional[str]]:
    """Validiert new_data, sichert die aktuelle config.json als Backup,
    schreibt new_data atomar nach config.json und übernimmt sie live ins
    laufende Programm. Wirft pydantic.ValidationError bei ungültigen Daten -
    dabei wird NICHTS geschrieben. Gibt (neue Config, Backup-Dateiname) zurück."""
    new_cfg = Config(**new_data)  # wirft ValidationError, wenn ungültig

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
