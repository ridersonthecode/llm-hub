"""Nutzerverwaltung fürs Dashboard: /dashboard/users (HTML, nur für Admins)
+ die zugehörigen JSON-Endpoints zum Anlegen/Zurücksetzen/Löschen von
users.json-Nutzern (siehe web_auth.py).

Admin-Rechte kommen ausschließlich von Linux-Systemnutzern mit Shell-Login
(siehe system_users.py) - die tauchen hier NICHT als Zeile auf, es gibt für
sie nichts zu verwalten (ihr Passwort ist ihr normales Linux-Passwort). Diese
Seite verwaltet nur die zusätzlichen, per Web-UI angelegten Nutzer ohne
Shell-Zugang (Rolle "user", siehe web_auth.authenticate)."""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from . import system_users
from .web_auth import (
    build_user_entry,
    read_users,
    render_nav_links_html,
    render_nav_user_html,
    username_conflicts_with_system_user,
    write_users,
)

router = APIRouter()
logger = logging.getLogger("llm_hub.users_dashboard")

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")
_MIN_PASSWORD_LEN = 8


def _require_admin(request: Request) -> None:
    if getattr(request.state, "role", None) != "admin":
        raise HTTPException(status_code=403, detail="Nur für Admins (Systemnutzer mit Shell-Login).")


@router.get("/dashboard/users", response_class=HTMLResponse)
async def users_dashboard_page(request: Request):
    if getattr(request.state, "role", None) != "admin":
        # Kein Nav-Link für Nicht-Admins (siehe render_nav_user_html), aber
        # eine direkt aufgerufene URL landet trotzdem sauber zurück statt auf
        # einer nackten 403-Seite.
        return RedirectResponse("/dashboard", status_code=302)
    html = USERS_DASHBOARD_HTML.replace("<!--NAV_USER-->", render_nav_user_html(request))
    html = html.replace("<!--NAV_LINKS-->", render_nav_links_html("users", i18n=False))
    return HTMLResponse(html)


@router.get("/dashboard/users/api/list")
async def list_users(request: Request):
    _require_admin(request)
    users = read_users()
    out = [
        {
            "username": username,
            "created_at": entry.get("created_at"),
            "created_by": entry.get("created_by"),
        }
        for username, entry in sorted(users.items())
    ]
    return {"users": out, "system_admins": system_users.list_admin_usernames()}


@router.post("/dashboard/users/api/create")
async def create_user(request: Request):
    _require_admin(request)
    body = await request.json()
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")

    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Ungültiger Benutzername (2-32 Zeichen, Buchstaben/Ziffern/._-).")
    if len(password) < _MIN_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail=f"Passwort zu kurz (mind. {_MIN_PASSWORD_LEN} Zeichen).")
    if username_conflicts_with_system_user(username):
        raise HTTPException(
            status_code=400,
            detail=f"'{username}' ist bereits ein Systemnutzer mit Shell-Login und dadurch automatisch Admin.",
        )

    users = read_users()
    if username in users:
        raise HTTPException(status_code=409, detail=f"Nutzer '{username}' existiert bereits.")
    users[username] = build_user_entry(password, created_by=request.state.username)
    write_users(users)
    logger.info("Nutzer '%s' angelegt von Admin '%s'.", username, request.state.username)
    return {"ok": True}


@router.post("/dashboard/users/api/reset-password")
async def reset_password(request: Request):
    _require_admin(request)
    body = await request.json()
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")

    if len(password) < _MIN_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail=f"Passwort zu kurz (mind. {_MIN_PASSWORD_LEN} Zeichen).")

    users = read_users()
    if username not in users:
        raise HTTPException(status_code=404, detail=f"Nutzer '{username}' existiert nicht.")
    # build_user_entry() zählt pw_version hoch -> jede bestehende Session
    # dieses Nutzers wird bei der nächsten Anfrage ungültig (siehe
    # web_auth._check_session).
    users[username] = build_user_entry(password, previous=users[username])
    write_users(users)
    logger.info("Passwort von Nutzer '%s' zurückgesetzt von Admin '%s'.", username, request.state.username)
    return {"ok": True}


@router.post("/dashboard/users/api/delete")
async def delete_user(request: Request):
    _require_admin(request)
    body = await request.json()
    username = str(body.get("username") or "").strip()

    users = read_users()
    if username not in users:
        raise HTTPException(status_code=404, detail=f"Nutzer '{username}' existiert nicht.")
    del users[username]
    write_users(users)
    logger.info("Nutzer '%s' gelöscht von Admin '%s'.", username, request.state.username)
    return {"ok": True}


USERS_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Hub – Nutzerverwaltung</title>
<style>
  :root {
    --bg:#f5f6f8; --panel:#ffffff; --panel-2:#eef0f4; --border:#dfe3ea;
    --text:#161922; --text-dim:#4b5363; --mono: "SF Mono", Consolas, "Liberation Mono", monospace;
    --accent:#2563eb; --good:#15803d; --warn:#b45309; --bad:#dc2626;
    --accent-bg:rgba(37,99,235,.10); --bad-bg:rgba(220,38,38,.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
      --text:#eef1f6; --text-dim:#a3acc2;
      --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
      --accent-bg:rgba(122,168,255,.15); --bad-bg:rgba(248,113,113,.15);
    }
  }
  :root[data-theme="dark"] {
    --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
    --text:#eef1f6; --text-dim:#a3acc2;
    --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
    --accent-bg:rgba(122,168,255,.15); --bad-bg:rgba(248,113,113,.15);
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,"Segoe UI",Roboto,sans-serif; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; }
  h2 { font-size:15px; margin:28px 0 10px; }
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; }
  .topbar-actions { display:flex; gap:8px; align-items:center; flex:0 0 auto; }
  .sub { color:var(--text-dim); font-size:13px; margin-bottom:8px; }
  .sub a { color:var(--accent); text-decoration:none; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
  th { color:var(--text-dim); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
  tr:last-child td { border-bottom:none; }
  .hint { color:var(--text-dim); font-size:12px; }
  .btn {
    background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:6px; padding:5px 10px; font-size:12px; cursor:pointer;
  }
  .btn:hover { border-color:var(--accent); }
  .btn.danger:hover { border-color:var(--bad); color:var(--bad); }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .row-actions { display:flex; gap:6px; }
  form.inline { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; margin-top:12px; }
  label { display:block; font-size:12px; color:var(--text-dim); margin-bottom:4px; }
  input {
    background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:6px; padding:7px 9px; font-size:13px;
  }
  .badge {
    display:inline-block; background:var(--accent-bg); color:var(--accent);
    border-radius:6px; padding:2px 8px; font-size:12px; margin:2px 6px 2px 0;
  }
  .msg { margin-top:10px; font-size:13px; }
  .msg.error { color:var(--bad); }
  .msg.ok { color:var(--good); }
  #theme-toggle {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; width:36px; height:36px; font-size:16px; cursor:pointer; flex:0 0 auto;
  }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>👤 Nutzerverwaltung</h1>
    </div>
    <div class="topbar-actions">
      <!--NAV_LINKS-->
      <button id="theme-toggle" title="Toggle theme">🌙</button>
      <!--NAV_USER-->
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Admins (Systemnutzer)</h2>
    <p class="hint">
      Jeder Linux-Nutzer dieser Maschine, der sich per Shell anmelden darf, ist automatisch Admin im Dashboard
      (Passwort = Linux-Passwort, geprüft per PAM). Dafür gibt es hier nichts zu verwalten.
    </p>
    <div id="system-admins"></div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2 style="margin-top:0">Dashboard-Nutzer (users.json)</h2>
    <p class="hint">Eigene Zugänge ohne Shell-Login auf dieser Maschine - von einem Admin hier angelegt.</p>
    <table>
      <thead><tr><th>Benutzername</th><th>Angelegt</th><th>Von</th><th>Aktionen</th></tr></thead>
      <tbody id="users-body"></tbody>
    </table>
    <div id="users-empty" class="hint" style="display:none;padding:10px 0;">Noch keine Nutzer angelegt.</div>

    <h2>Neuen Nutzer anlegen</h2>
    <form class="inline" id="create-form">
      <div>
        <label for="new-username">Benutzername</label>
        <input type="text" id="new-username" required autocomplete="off">
      </div>
      <div>
        <label for="new-password">Passwort</label>
        <input type="password" id="new-password" required autocomplete="new-password">
      </div>
      <button type="submit" class="btn primary">+ Anlegen</button>
    </form>
    <div id="create-msg" class="msg"></div>
  </div>

<script>
function esc(s) { return (s ?? "").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function fmtDate(ts) {
  if (!ts) return "–";
  return new Date(ts * 1000).toLocaleString();
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
  let data = null;
  try { data = await res.json(); } catch (e) {}
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
}

async function refresh() {
  const data = await api("/dashboard/users/api/list");
  const admins = document.getElementById("system-admins");
  admins.innerHTML = data.system_admins.map(u => `<span class="badge">${esc(u)}</span>`).join("") || `<span class="hint">Keine gefunden.</span>`;

  const body = document.getElementById("users-body");
  const empty = document.getElementById("users-empty");
  if (!data.users.length) {
    body.innerHTML = "";
    empty.style.display = "block";
  } else {
    empty.style.display = "none";
    body.innerHTML = data.users.map(u => `
      <tr>
        <td>${esc(u.username)}</td>
        <td>${fmtDate(u.created_at)}</td>
        <td>${esc(u.created_by || "–")}</td>
        <td class="row-actions">
          <button class="btn" data-reset="${esc(u.username)}">Passwort zurücksetzen</button>
          <button class="btn danger" data-delete="${esc(u.username)}">Löschen</button>
        </td>
      </tr>`).join("");
  }
}

document.getElementById("create-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("create-msg");
  msg.textContent = ""; msg.className = "msg";
  const username = document.getElementById("new-username").value.trim();
  const password = document.getElementById("new-password").value;
  try {
    await api("/dashboard/users/api/create", { method: "POST", body: JSON.stringify({ username, password }) });
    document.getElementById("create-form").reset();
    msg.textContent = `Nutzer '${username}' angelegt.`; msg.className = "msg ok";
    await refresh();
  } catch (err) {
    msg.textContent = err.message; msg.className = "msg error";
  }
});

document.getElementById("users-body").addEventListener("click", async (e) => {
  const resetUser = e.target.dataset.reset;
  const deleteUser = e.target.dataset.delete;
  if (resetUser) {
    const password = prompt(`Neues Passwort für '${resetUser}' (mind. 8 Zeichen):`);
    if (!password) return;
    try {
      await api("/dashboard/users/api/reset-password", { method: "POST", body: JSON.stringify({ username: resetUser, password }) });
      alert("Passwort geändert.");
    } catch (err) { alert(err.message); }
  } else if (deleteUser) {
    if (!confirm(`Nutzer '${deleteUser}' wirklich löschen?`)) return;
    try {
      await api("/dashboard/users/api/delete", { method: "POST", body: JSON.stringify({ username: deleteUser }) });
      await refresh();
    } catch (err) { alert(err.message); }
  }
});

function currentTheme() {
  return document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function updateToggleIcon(theme) { document.getElementById("theme-toggle").textContent = theme === "dark" ? "☀️" : "🌙"; }
(function initTheme() {
  const saved = localStorage.getItem("vllm_dashboard_theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  updateToggleIcon(currentTheme());
})();
document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("vllm_dashboard_theme", next);
  updateToggleIcon(next);
});

refresh().catch(err => alert(err.message));
</script>
</body>
</html>"""
