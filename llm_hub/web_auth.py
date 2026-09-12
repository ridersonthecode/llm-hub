"""Website-Login fürs Dashboard: echte, signierte Session-Cookies statt HTTP
Basic Auth (Chat vom 2026-09-06: 'eigene Anmeldeseite, kein Basic Auth mehr,
richtige Sessions').

Zwei Nutzerklassen:
- Linux-Systemnutzer mit echter Shell-Login-Berechtigung (siehe
  system_users.py, z.B. root oder mwagner) -> Rolle "admin", Passwort wird
  live per PAM geprüft. Darf u.a. die Nutzerverwaltung (/dashboard/users)
  nutzen.
- users.json-Nutzer, angelegt übers Dashboard (siehe users_dashboard.py) oder
  per CLI-Fallback (manage_users.py) -> Rolle "user", eigener PBKDF2-Hash.

Kein Server-seitiger Session-Store (kein Datenbank-Zwang für dieses Projekt):
das Cookie selbst trägt Nutzername+Rolle+Passwort-Version+Ablauf, signiert per
HMAC-SHA256 mit einem lokal erzeugten Secret (.session_secret). Bei jeder
Anfrage prüft die Middleware zusätzlich billig (kein PBKDF2/PAM, nur ein
dict-/pwd-Lookup) nach, dass die im Cookie behauptete Rolle noch aktuell ist -
ein entfernter users.json-Nutzer oder ein Systemnutzer ohne Shell mehr fliegt
so spätestens bei der nächsten Anfrage raus, ganz ohne Session-Tabelle.

Komplett unabhängig von der optionalen API-Key-Prüfung in auth.py: die
betrifft ausschließlich den OpenAI-kompatiblen /v1-Proxy (für Tools wie
VS Code/Copilot, die keinen Login-Dialog anzeigen können) und bleibt
standardmäßig aus. Dieses Modul schützt umgekehrt genau das, was die
API-Key-Prüfung bewusst ausnimmt: die Web-Oberfläche selbst.

users.json anlegen/verwalten: normalerweise übers Dashboard
(/dashboard/users, nur für Admins), alternativ per CLI als Fallback (siehe
manage_users.py) - z.B. falls sich mal niemand mehr einloggen kann. Ohne
users.json (oder mit einer leeren Datei) UND ohne qualifizierenden
Systemnutzer bleibt die Website wie bisher offen."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import json
import logging
import os
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from starlette.datastructures import Headers
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from . import system_users
from .config import PROJECT_ROOT

USERS_PATH = Path(os.environ.get("LLM_HUB_USERS", PROJECT_ROOT / "users.json"))
SESSION_SECRET_PATH = Path(os.environ.get("LLM_HUB_SESSION_SECRET", PROJECT_ROOT / ".session_secret"))

# Bewusst deutlich über den früher üblichen 100k (siehe OWASP-Empfehlung für
# PBKDF2-HMAC-SHA256, Stand 2023) - das Dashboard läuft lokal auf derselben
# GPU-Maschine, die zusätzlichen paar Millisekunden pro Passwort-Prüfung
# fallen dort nicht ins Gewicht. Anders als bei der alten Basic-Auth-Variante
# läuft das jetzt nur noch EINMAL pro Login, nicht mehr bei jeder Anfrage
# (siehe SessionAuthMiddleware, die nur noch die Cookie-Signatur prüft).
_PBKDF2_ITERATIONS = 260_000

SESSION_COOKIE_NAME = "llm_hub_session"
SESSION_MAX_AGE_SECONDS = 14 * 24 * 3600  # 14 Tage

# Von der Website-Anmeldung ausgenommen: die eigentliche(n) API(s), die
# externe Tools OHNE Zugangsdaten nutzen sollen (siehe Chat-Anfrage "die api
# soll nicht mit zugangsdaten geschützt werden") - der OpenAI-kompatible
# Proxy (/v1, genutzt z.B. von VS Code/Copilot), das Ollama-Kompatibilitäts-
# layer (/api, siehe ollama_compat.py) und der MCP-Server (/mcp). /health ist
# für einfache Monitoring-Checks (z.B. systemd/uptime-Skripte) gedacht, die
# ebenfalls keinen Login-Dialog beantworten können.
EXEMPT_PREFIXES = ("/v1/", "/api/", "/mcp")
# /login und /logout sind ganz normale Teile der Web-Oberfläche - aber die
# einzigen zwei, die man OHNE gültige Session erreichen muss (sonst Redirect-
# Schleife: keine Session -> Redirect zu /login -> /login selbst verlangt
# wieder eine Session -> ...).
NO_SESSION_REQUIRED = {"/login", "/logout"}

NAV_LINK_STYLE = (
    "display:inline-flex;align-items:center;background:var(--panel);"
    "border:1px solid var(--border);color:var(--text);text-decoration:none;"
    "border-radius:8px;height:36px;padding:0 12px;font-size:13px;box-sizing:border-box;"
)

logger = logging.getLogger("llm_hub.web_auth")


def _is_exempt(path: str) -> bool:
    if path == "/health" or path in NO_SESSION_REQUIRED:
        return True
    return path.startswith(EXEMPT_PREFIXES)


# --- Passwort-Hashing (users.json) -----------------------------------------

def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Gibt (salt_hex, hash_hex) zurück - genutzt von build_user_entry() beim
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
    return hmac.compare_digest(actual, expected)


def build_user_entry(password: str, *, previous: Optional[dict] = None, created_by: Optional[str] = None) -> dict:
    """Baut einen neuen bzw. aktualisiert einen bestehenden users.json-Eintrag:
    frischer Salt+Hash, hochgezählte 'pw_version'. Die Versionsnummer landet
    im Session-Cookie (siehe create_session_token) - eine bestehende Session
    mit alter Versionsnummer wird bei der nächsten Anfrage abgelehnt, eine
    Passwortänderung loggt alte Sessions dieses Nutzers also automatisch aus.
    created_at/created_by bleiben bei einer reinen Passwortänderung erhalten."""
    salt_hex, hash_hex = hash_password(password)
    previous = previous or {}
    entry = {
        "salt": salt_hex,
        "hash": hash_hex,
        "pw_version": int(previous.get("pw_version") or 0) + 1,
        "created_at": previous.get("created_at") or time.time(),
    }
    creator = previous.get("created_by") or created_by
    if creator:
        entry["created_by"] = creator
    return entry


def username_conflicts_with_system_user(username: str) -> bool:
    """True, wenn 'username' bereits ein Linux-Systemnutzer mit Shell-Login
    ist (siehe system_users.is_shell_login_user) - ein users.json-Eintrag mit
    demselben Namen würde diesen beim Login nie erreichen (authenticate()
    prüft Systemnutzer zuerst) und wird deshalb schon beim Anlegen abgelehnt,
    siehe users_dashboard.py."""
    return system_users.is_shell_login_user(username)


# --- users.json I/O ----------------------------------------------------------

def read_users() -> dict:
    """Liest users.json ungecached direkt von der Platte - genutzt von
    manage_users.py/users_dashboard.py, wo Änderungen sofort sichtbar sein
    müssen. Für die Middleware selbst siehe _load_users_cached() weiter unten."""
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
    SessionAuthMiddleware), ein Reread bei jedem einzelnen Request wäre
    unnötiger Disk-I/O für eine Datei, die sich praktisch nie ändert."""
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
            logger.exception("users.json konnte nicht gelesen werden - users.json-Logins bleiben bis zur Reparatur deaktiviert.")
            _users_cache = {}
        _users_mtime = mtime
    return _users_cache


# --- Login (Systemnutzer per PAM ODER users.json) ---------------------------

def authenticate(username: str, password: str) -> Optional[dict]:
    """Zentrale Login-Prüfung: erst Systemnutzer/PAM (-> Rolle admin), sonst
    users.json (-> Rolle user). Blockierend (PBKDF2 bzw. PAM/unix_chkpwd-
    Subprozess) - von der Login-Route per asyncio.to_thread() aufgerufen,
    NICHT vom Request-Hot-Path (den prüft SessionAuthMiddleware nur noch
    anhand des signierten Cookies)."""
    if not username or not password:
        return None
    if system_users.is_shell_login_user(username):
        # Kein Fallback auf users.json bei falschem Systempasswort: ein
        # gleichnamiger users.json-Eintrag würde diesen Zweig ohnehin nie
        # erreichen und wird deshalb schon beim Anlegen verhindert (siehe
        # username_conflicts_with_system_user).
        if system_users.authenticate_system_user(username, password):
            return {"username": username, "role": "admin", "pw_version": 0}
        return None
    entry = read_users().get(username)
    if not entry or not _verify_password(password, entry.get("salt", ""), entry.get("hash", "")):
        return None
    return {"username": username, "role": "user", "pw_version": int(entry.get("pw_version") or 1)}


# --- Signierte Session-Cookies -----------------------------------------------

_session_secret: Optional[bytes] = None


def _load_or_create_session_secret() -> bytes:
    """Lädt das lokale HMAC-Secret fürs Signieren der Session-Cookies, legt
    beim allerersten Start eins an (32 zufällige Bytes, hex-kodiert, 0600) -
    ein Neustart des Prozesses invalidiert damit KEINE bestehenden Sessions
    (anders als ein In-Memory-Secret), ein manuelles Löschen der Datei dagegen
    schon (loggt alle Nutzer aus)."""
    global _session_secret
    if _session_secret is not None:
        return _session_secret
    try:
        _session_secret = bytes.fromhex(SESSION_SECRET_PATH.read_text().strip())
        return _session_secret
    except FileNotFoundError:
        pass
    except ValueError:
        logger.warning("%s enthielt keinen gültigen Hex-Schlüssel - erzeuge neuen (alle Sessions werden ungültig).", SESSION_SECRET_PATH)
    secret = os.urandom(32)
    try:
        fd = os.open(SESSION_SECRET_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secret.hex())
        _session_secret = secret
    except FileExistsError:
        # Zwischen Lesen und Schreiben hat ein paralleler Request die Datei
        # bereits angelegt (Race beim allerersten Start) - deren Wert gilt.
        _session_secret = bytes.fromhex(SESSION_SECRET_PATH.read_text().strip())
    return _session_secret


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_session_token(username: str, role: str, pw_version: int = 0, max_age: int = SESSION_MAX_AGE_SECONDS) -> str:
    """Signiertes, zustandsloses Session-Token (HMAC-SHA256) - kein Server-
    seitiger Session-Store nötig. 'pw_version' erlaubt trotzdem gezieltes
    Invalidieren: ändert sich der gespeicherte Passwort-Hash eines
    users.json-Nutzers, wird eine alte Session mit veralteter Versionsnummer
    bei der nächsten Anfrage abgelehnt (siehe SessionAuthMiddleware)."""
    payload = {"u": username, "r": role, "v": pw_version, "exp": int(time.time()) + max_age}
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = hmac.new(_load_or_create_session_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_session_token(token: str) -> Optional[dict]:
    """Prüft Signatur + Ablauf, gibt bei Erfolg {'username','role','pw_version'}
    zurück, sonst None. Prüft NICHT, ob der Nutzer noch existiert/noch dieselbe
    Rolle hat - das übernimmt SessionAuthMiddleware zusätzlich pro Anfrage."""
    try:
        body, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    expected_sig = hmac.new(_load_or_create_session_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(body))
    except Exception:
        return None
    if not isinstance(payload, dict) or float(payload.get("exp", 0)) < time.time():
        return None
    username, role = payload.get("u"), payload.get("r")
    if not isinstance(username, str) or role not in ("admin", "user"):
        return None
    return {"username": username, "role": role, "pw_version": int(payload.get("v") or 0)}


def _parse_cookie_header(value: str) -> dict[str, str]:
    jar: SimpleCookie = SimpleCookie()
    jar.load(value)
    return {k: morsel.value for k, morsel in jar.items()}


def _check_session(headers: Headers) -> Optional[dict]:
    """Kern der Session-Prüfung, von SessionAuthMiddleware UND der /login-Route
    genutzt (letztere, um einen bereits eingeloggten Nutzer nicht nochmal das
    Formular sehen zu lassen)."""
    cookie_header = headers.get("cookie")
    if not cookie_header:
        return None
    token = _parse_cookie_header(cookie_header).get(SESSION_COOKIE_NAME)
    if not token:
        return None
    claim = verify_session_token(token)
    if claim is None:
        return None
    if claim["role"] == "admin":
        if not system_users.is_shell_login_user(claim["username"]):
            return None
        return claim
    entry = _load_users_cached().get(claim["username"])
    if not entry or int(entry.get("pw_version") or 1) != claim["pw_version"]:
        return None
    return claim


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    # Secure-Flag nur bei tatsächlichem HTTPS-Zugriff setzen - sonst würde das
    # Cookie über pures HTTP (Normalfall: internes Netz ohne eigenes
    # TLS-Zertifikat) nie zurückgeschickt und der Login liefe ins Leere.
    response.set_cookie(
        SESSION_COOKIE_NAME, token, max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True, samesite="lax", secure=request.url.scheme == "https", path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


class SessionAuthMiddleware:
    """Schützt die Web-Oberfläche (Dashboard-Seiten + deren Backend-Endpoints,
    siehe EXEMPT_PREFIXES) per Session-Cookie. Reine ASGI-Middleware statt
    @app.middleware("http") - aus demselben Grund wie ApiKeyMiddleware in
    auth.py (BaseHTTPMiddleware bricht Streaming-Responses, siehe dortigen
    Docstring); hier zusätzlich relevant, weil sie auch WebSocket-Handshakes
    abfangen muss, was BaseHTTPMiddleware gar nicht unterstützt.

    Bei gültiger Session landen 'username'/'role' in scope['state'] (Starlettes
    Konvention für Request.state) - Routen können das per request.state.role
    lesen, siehe z.B. users_dashboard.py."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if _is_exempt(path):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        identity = _check_session(headers)

        if identity is not None:
            scope.setdefault("state", {})
            scope["state"]["username"] = identity["username"]
            scope["state"]["role"] = identity["role"]
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # Vor dem Accept ablehnen (kein 'websocket.accept' gesendet) -
            # Uvicorn beantwortet den Handshake dann mit HTTP 403, statt die
            # Verbindung erst anzunehmen und danach wieder zu schließen.
            await send({"type": "websocket.close", "code": 4401})
            return

        if "text/html" in headers.get("accept", ""):
            next_path = path
            if scope.get("query_string"):
                next_path += "?" + scope["query_string"].decode("latin-1")
            response = RedirectResponse("/login?next=" + quote(next_path, safe=""), status_code=302)
        else:
            # Fetch/XHR-Aufrufe der Dashboard-JS (Accept ist dort nicht
            # "text/html") bekommen ein reguläres 401 statt eines Redirects,
            # den fetch() ohnehin nur als HTML-Body sähe.
            response = JSONResponse({"error": "Anmeldung erforderlich. Bitte auf /login einloggen."}, status_code=401)
        _clear_session_cookie(response)
        await response(scope, receive, send)


# --- Login-/Logout-Seiten -----------------------------------------------------

auth_router = APIRouter()

_LOGIN_HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Hub – Anmeldung</title>
<style>
  :root {
    --bg:#f5f6f8; --panel:#ffffff; --panel-2:#eef0f4; --border:#dfe3ea;
    --text:#161922; --text-dim:#4b5363; --accent:#2563eb; --bad:#dc2626; --bad-bg:rgba(220,38,38,.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
      --text:#eef1f6; --text-dim:#a3acc2; --accent:#7aa8ff; --bad:#f87171; --bad-bg:rgba(248,113,113,.15);
    }
  }
  :root[data-theme="dark"] {
    --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
    --text:#eef1f6; --text-dim:#a3acc2; --accent:#7aa8ff; --bad:#f87171; --bad-bg:rgba(248,113,113,.15);
  }
  * { box-sizing:border-box; }
  body {
    margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
    background:var(--bg); color:var(--text); font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  form {
    width:320px; background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:28px; box-shadow:0 8px 24px rgba(0,0,0,.08);
  }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:var(--text-dim); font-size:13px; margin-bottom:20px; }
  label { display:block; font-size:12px; color:var(--text-dim); margin:14px 0 4px; }
  input {
    width:100%; background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:9px 10px; font-size:14px;
  }
  input:focus { outline:2px solid var(--accent); outline-offset:1px; }
  button {
    width:100%; margin-top:20px; background:var(--accent); border:none; color:#fff;
    border-radius:8px; padding:10px; font-size:14px; font-weight:600; cursor:pointer;
  }
  button:hover { opacity:.92; }
  .error {
    margin-top:14px; background:var(--bad-bg); color:var(--bad); border-radius:8px;
    padding:8px 10px; font-size:13px;
  }
</style>
</head>
<body>
  <form method="post" action="/login">
    <h1>LLM Hub</h1>
    <div class="sub">Anmeldung erforderlich</div>
    <label for="username">Benutzername</label>
    <input type="text" id="username" name="username" autocomplete="username" autofocus required>
    <label for="password">Passwort</label>
    <input type="password" id="password" name="password" autocomplete="current-password" required>
    <input type="hidden" name="next" value="__NEXT__">
    __ERROR__
    <button type="submit">Anmelden</button>
  </form>
<script>
  const saved = localStorage.getItem("vllm_dashboard_theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
</script>
</body>
</html>"""


def _render_login_page(next_path: str, error: Optional[str]) -> str:
    error_html = '<div class="error">Benutzername oder Passwort falsch.</div>' if error else ""
    return _LOGIN_HTML.replace("__NEXT__", html.escape(next_path, quote=True)).replace("__ERROR__", error_html)


def _safe_next_path(value: Optional[str]) -> str:
    """Verhindert einen offenen Redirect über den 'next'-Parameter: nur
    Pfade innerhalb dieser Website sind erlaubt, kein 'https://anderes-ziel'
    und kein protokoll-relatives '//anderes-ziel'."""
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/dashboard"


@auth_router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    if _check_session(Headers(scope=request.scope)) is not None:
        return RedirectResponse(_safe_next_path(request.query_params.get("next")), status_code=302)
    next_path = _safe_next_path(request.query_params.get("next"))
    return HTMLResponse(_render_login_page(next_path, request.query_params.get("error")))


@auth_router.post("/login", include_in_schema=False)
async def login_submit(request: Request):
    form = await request.form()
    username = str(form.get("username") or "").strip()
    password = str(form.get("password") or "")
    next_path = _safe_next_path(str(form.get("next") or ""))
    identity = await asyncio.to_thread(authenticate, username, password)
    if identity is None:
        logger.warning("Fehlgeschlagener Dashboard-Login für Benutzername '%s'.", username)
        return RedirectResponse(f"/login?error=1&next={quote(next_path, safe='')}", status_code=302)
    logger.info("Dashboard-Login erfolgreich: '%s' (Rolle %s).", identity["username"], identity["role"])
    token = create_session_token(identity["username"], identity["role"], identity["pw_version"])
    response = RedirectResponse(next_path, status_code=302)
    set_session_cookie(response, request, token)
    return response


@auth_router.get("/logout", include_in_schema=False)
async def logout():
    response = RedirectResponse("/login", status_code=302)
    _clear_session_cookie(response)
    return response


# --- Topbar-Snippet für alle Dashboard-Seiten --------------------------------

# Die Cross-Links zwischen den Dashboard-Seiten selbst (Dashboard/Config/
# Costs/RAG/Conversations) - Reihenfolge = Reihenfolge in der Topbar.
# (key, Ziel-URL, i18n-Key, Default-Text (Englisch, siehe languages/*.json),
#  Text ohne i18n-System (siehe users_dashboard.py, das ist Deutsch-only).)
_NAV_ITEMS = (
    ("dashboard", "/dashboard", "nav.dashboardLink", "← Dashboard", "← Dashboard"),
    ("config", "/dashboard/config", "nav.configLink", "⚙️ Config →", "⚙️ Config →"),
    ("costs", "/dashboard/costs", "nav.costsLink", "💰 Costs →", "💰 Kosten →"),
    ("rag", "/dashboard/rag", "nav.ragLink", "RAG →", "RAG →"),
    ("conversations", "/dashboard/conversations", "nav.conversationsLink", "🗨️ Conversations →", "🗨️ Konversationen →"),
)


def render_nav_links_html(active: str, *, i18n: bool = True) -> str:
    """Gemeinsame Cross-Links zwischen den Dashboard-Seiten (siehe
    _NAV_ITEMS) fürs Topbar jeder Seite. Wird per Platzhalter
    '<!--NAV_LINKS-->' in jedes *_dashboard.py-Template eingesetzt (siehe
    render_nav_user_html unten für das Pendant mit Nutzer/Logout).

    `active` ist die aktuell angezeigte Seite (z.B. "config") und wird nicht
    verlinkt - so landet niemand per Klick auf der Seite, auf der er schon
    ist. Seiten, die (noch) keinen eigenen Eintrag haben (z.B. "chat", siehe
    chat_dashboard.py - bewusst nicht in der Hauptnavigation verlinkt),
    übergeben einfach ihren eigenen Namen als `active`; taucht der in
    _NAV_ITEMS gar nicht auf, werden schlicht alle Links angezeigt.

    `i18n=False` für Seiten ohne Sprachumschaltung (aktuell nur
    users_dashboard.py): dann werden feste deutsche Texte statt
    data-i18n-Attributen ausgegeben."""
    parts = []
    for key, href, i18n_key, en_label, de_label in _NAV_ITEMS:
        if key == active:
            continue
        if i18n:
            parts.append(f'<a href="{href}" data-i18n="{i18n_key}" style="{NAV_LINK_STYLE}">{en_label}</a>')
        else:
            parts.append(f'<a href="{href}" style="{NAV_LINK_STYLE}">{de_label}</a>')
    return "".join(parts)


def render_nav_user_html(request: Request) -> str:
    """Kleines HTML-Snippet fürs Topbar jeder Dashboard-Seite: eingeloggter
    Nutzer + Rolle, Link zur Nutzerverwaltung (nur für Admins) und Logout.
    Wird per Platzhalter '<!--NAV_USER-->' in jedes *_dashboard.py-Template
    eingesetzt (siehe z.B. dashboard.py)."""
    username = getattr(request.state, "username", None)
    role = getattr(request.state, "role", None)
    if not username:
        return ""
    users_link = (
        f'<a href="/dashboard/users" id="users-link" style="{NAV_LINK_STYLE}">👤 Users →</a>'
        if role == "admin" else ""
    )
    role_label = "Admin" if role == "admin" else "User"
    return (
        f'{users_link}'
        f'<span title="{html.escape(role_label)}" '
        f'style="color:var(--text-dim);font-size:13px;align-self:center;padding:0 4px;white-space:nowrap;">'
        f'{html.escape(username)} · {html.escape(role_label)}</span>'
        f'<a href="/logout" id="logout-link" style="{NAV_LINK_STYLE}">Logout</a>'
    )
