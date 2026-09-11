"""Linux-Systemnutzer als 'Master'/Admin fürs Dashboard-Login (siehe web_auth.py).

Idee (Chat vom 2026-09-06): Admin-Rechte im Dashboard bekommt automatisch
jeder lokale Linux-Nutzer, der sich auch per Shell an dieser Maschine anmelden
darf - egal ob root oder ein normaler Nutzer wie 'mwagner'. Alle anderen
Dashboard-Nutzer sind eigene, per Web-UI angelegte Accounts ohne Shell-Zugang
(siehe users.json/web_auth.py) und bekommen nie Admin-Rechte.

Passwortprüfung läuft über PAM (pam_unix), NICHT durch direktes Lesen von
/etc/shadow - der llm-hub-Prozess läuft unprivilegiert (siehe llm-hub.service,
User=mwagner) und hat gar keinen Lesezugriff auf /etc/shadow (0640, root:shadow).
PAM braucht das auch nicht: pam_unix.so delegiert die eigentliche Prüfung an
/usr/sbin/unix_chkpwd, ein kleines setgid-shadow-Hilfsprogramm genau für diesen
Zweck (Login-Manager, Screensaver, ... - alles unprivilegierte Prozesse, die
trotzdem ein Systempasswort prüfen können müssen). Deshalb funktioniert das
hier ganz ohne Root-Rechte für llm-hub selbst.

Benötigt das Paket 'python-pam' (pip install python-pam) - siehe README.md."""
from __future__ import annotations

import logging
import os
import pwd
from pathlib import Path

logger = logging.getLogger("llm_hub.system_users")

# PAM-Service, gegen den authentifiziert wird - 'login' existiert auf jeder
# gängigen Distro (siehe /etc/pam.d/login) und reicht für eine reine
# Passwortprüfung aus. Per Env überschreibbar, falls eine Installation mal nur
# eine minimale PAM-Konfiguration ohne 'login'-Service mitbringt.
PAM_SERVICE = os.environ.get("LLM_HUB_PAM_SERVICE", "login")

_SHELLS_PATH = Path("/etc/shells")

_pam_import_warned = False


def _valid_login_shells() -> set[str]:
    """Liest /etc/shells (Liste 'echter' interaktiver Shells) - Systemkonten
    mit z.B. /usr/sbin/nologin oder /bin/false landen bewusst NICHT hier drin
    und gelten damit nicht als 'darf sich per Shell anmelden'."""
    try:
        with open(_SHELLS_PATH, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and not line.startswith("#")}
    except FileNotFoundError:
        logger.warning("%s existiert nicht - Fallback auf /bin/bash, /bin/sh.", _SHELLS_PATH)
        return {"/bin/bash", "/bin/sh"}


def is_shell_login_user(username: str) -> bool:
    """True, wenn 'username' ein lokaler Systemnutzer mit einer echten,
    interaktiven Login-Shell ist (siehe /etc/shells) - genau diese Nutzer
    werden im Dashboard automatisch zu Admins, siehe Modul-Docstring."""
    try:
        entry = pwd.getpwnam(username)
    except KeyError:
        return False
    return entry.pw_shell in _valid_login_shells()


def list_admin_usernames() -> list[str]:
    """Alle aktuell qualifizierenden Systemnutzer - nur fürs Anzeigen im
    Dashboard gedacht (Transparenz: 'diese Accounts sind automatisch Admin'),
    NICHT für Auth-Entscheidungen selbst (die laufen über is_shell_login_user())."""
    shells = _valid_login_shells()
    return sorted({u.pw_name for u in pwd.getpwall() if u.pw_shell in shells})


def authenticate_system_user(username: str, password: str) -> bool:
    """Prüft das Systempasswort per PAM. Gibt bei fehlender 'pam'-Bibliothek
    oder sonstigen PAM-Fehlern False zurück (mit Log-Eintrag) statt zu
    crashen - ein falsch konfiguriertes PAM soll nicht den kompletten
    Dashboard-Login lahmlegen, sondern nur den Systemnutzer-Login-Weg."""
    global _pam_import_warned
    try:
        import pam
    except ImportError:
        if not _pam_import_warned:
            logger.error(
                "Paket 'python-pam' fehlt - Systemnutzer-Login (Admin) ist deaktiviert. "
                "Installieren mit: pip install python-pam"
            )
            _pam_import_warned = True
        return False
    try:
        return bool(pam.pam().authenticate(username, password, service=PAM_SERVICE))
    except Exception:
        logger.exception("PAM-Authentifizierung für Systemnutzer '%s' technisch fehlgeschlagen.", username)
        return False
