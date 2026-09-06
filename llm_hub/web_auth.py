"""Website-Login fürs Dashboard: einfache Benutzername/Passwort-Prüfung per
HTTP Basic Auth gegen users.json (kein Datenbank, kein Session-Handling
nötig - der Browser cached die Basic-Auth-Credentials selbst für die gesamte
Origin, inklusive Fetch-Aufrufe und WebSocket-Handshakes).

Komplett unabhängig von der optionalen API-Key-Prüfung in auth.py: die
betrifft ausschließlich den OpenAI-kompatiblen /v1-Proxy (für Tools wie
VS Code/Copilot, die keinen Basic-Auth-Dialog anzeigen können) und bleibt
standardmäßig aus. Dieses Modul schützt umgekehrt genau das, was die
API-Key-Prüfung bewusst ausnimmt: die Web-Oberfläche selbst.

users.json anlegen/verwalten: siehe manage_users.py ('python -m
llm_hub.manage_users add <username>'). Ohne users.json (oder mit einer
leeren Datei) bleibt die Website wie bisher offen - genau wie ApiKeyMiddleware
bei api_key.enabled=false."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Optional

from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import PROJECT_ROOT

USERS_PATH = Path(os.environ.get("LLM_HUB_USERS", PROJECT_ROOT / "users.json"))

# Bewusst deutlich über den früher üblichen 100k (siehe OWASP-Empfehlung für
# PBKDF2-HMAC-SHA256, Stand 2023) - das Dashboard läuft lokal auf derselben
# GPU-Maschine, die zusätzlichen paar Millisekunden pro Login fallen dort
# nicht ins Gewicht.
_PBKDF2_ITERATIONS = 260_000
_REALM = "LLM Hub"

# Von der Website-Anmeldung ausgenommen: die eigentliche(n) API(s), die
# externe Tools OHNE Zugangsdaten nutzen sollen (siehe Chat-Anfrage "die api
# soll nicht mit zugangsdaten geschützt werden") - der OpenAI-kompatible
# Proxy (/v1, genutzt z.B. von VS Code/Copilot), das Ollama-Kompatibilitäts-
# layer (/api, siehe ollama_compat.py) und der MCP-Server (/mcp). /health ist
# für einfache Monitoring-Checks (z.B. systemd/uptime-Skripte) gedacht, die
# ebenfalls keinen Login-Dialog beantworten können.
#
# ALLES andere - die Dashboard-Seiten selbst sowie ihre Backend-Endpoints
# (/models, /config, /costs, /conversations, /rag, /requests, /static) -
# verlangt bei konfigurierten Nutzern ein gültiges Login.
EXEMPT_PREFIXES = ("/v1/", "/api/", "/mcp")

logger = logging.getLogger("llm_hub.web_auth")


def _is_exempt(path: str) -> bool:
    if path == "/health":
        return True
    return path.startswith(EXEMPT_PREFIXES)


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Gibt (salt_hex, hash_hex) zurück - genutzt von manage_users.py beim
    Anlegen/Ändern eines Nutzers. salt=None erzeugt ein frisches, zufälliges
    Salt (Normalfall); ein übergebenes salt ist nur für Tests gedacht."""
    salt = salt if salt is not None else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    # Konstante Laufzeit gegen Timing-Angriffe, wie beim vorhandenen API-Key-
    # Vergleich (siehe auth.py - dort reicht der übliche String-Vergleich von
    # Starlettes Headers-Objekt allerdings schon aus, da der API-Key selbst
    # nicht gehasht wird).
    return hmac.compare_digest(actual, expected)


def read_users() -> dict:
    """Liest users.json ungecached direkt von der Platte - genutzt von
    manage_users.py, wo Änderungen sofort sichtbar sein müssen. Für die
    Middleware selbst siehe _load_users_cached() weiter unten."""
    if not USERS_PATH.exists():
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_users(users: dict) -> None:
    """Schreibt users.json atomar (wie config._atomic_write_json) und mit
    restriktiven Rechten (0600) - die Datei enthält Salt+PBKDF2-Hash je
    Nutzer, kein Klartext-Passwort, aber die Hashes sind trotzdem schützens-
    wert (Offline-Bruteforce bei Kompromittierung)."""
    tmp = USERS_PATH.with_suffix(USERS_PATH.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, USERS_PATH)


_users_cache: dict = {}
_users_mtime: Optional[float] = None


def _load_users_cached() -> dict:
    """Wie read_users(), aber mit Invalidierung nur bei geänderter mtime -
    users.json wird bei JEDEM HTTP-/WebSocket-Request geprüft (siehe
    WebAuthMiddleware), ein Reread bei jedem einzelnen Request wäre unnötiger
    Disk-I/O für eine Datei, die sich praktisch nie ändert."""
    global _users_cache, _users_mtime
    try:
        mtime = USERS_PATH.stat().st_mtime
    except FileNotFoundError:
        _users_cache, _users_mtime = {}, None
        return _users_cache
    if mtime != _users_mtime:
        try:
            with open(USERS_PATH, "r", encoding="utf-8") as f:
                _users_cache = json.load(f)
        except Exception:
            logger.exception("users.json konnte nicht gelesen werden - Website-Login bleibt bis zur Reparatur deaktiviert.")
            _users_cache = {}
        _users_mtime = mtime
    return _users_cache


def _check_credentials(header_value: Optional[str]) -> bool:
    users = _load_users_cached()
    if not users:
        return True  # keine Nutzer konfiguriert -> Website-Login deaktiviert
    if not header_value or not header_value.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header_value[len("Basic "):]).decode("utf-8")
        username, _, password = decoded.partition(":")
    except Exception:
        return False
    entry = users.get(username)
    if not entry:
        return False
    return _verify_password(password, entry.get("salt", ""), entry.get("hash", ""))


class WebAuthMiddleware:
    """Schützt die Web-Oberfläche (Dashboard-Seiten + deren Backend-Endpoints,
    siehe EXEMPT_PREFIXES) per HTTP Basic Auth gegen users.json. Reine ASGI-
    Middleware statt @app.middleware("http") - aus demselben Grund wie
    ApiKeyMiddleware in auth.py (BaseHTTPMiddleware bricht Streaming-
    Responses, siehe dortigen Docstring); hier zusätzlich relevant, weil sie
    auch WebSocket-Handshakes abfangen muss, was BaseHTTPMiddleware gar nicht
    unterstützt."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if _is_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if _check_credentials(headers.get("authorization")):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # Vor dem Accept ablehnen (kein 'websocket.accept' gesendet) -
            # Uvicorn beantwortet den Handshake dann mit HTTP 403, statt die
            # Verbindung erst anzunehmen und danach wieder zu schließen.
            await send({"type": "websocket.close", "code": 4401})
            return

        response = Response(
            "Anmeldung erforderlich. Bitte Benutzername/Passwort eingeben "
            "(siehe 'python -m llm_hub.manage_users').",
            status_code=401,
            headers={"WWW-Authenticate": f'Basic realm="{_REALM}"'},
        )
        await response(scope, receive, send)
