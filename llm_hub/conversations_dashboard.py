"""Konversations-Log-Seite: /dashboard/conversations (HTML), /dashboard/
conversations/status (JSON-Snapshot), /dashboard/conversations/ws (WebSocket,
Live-Push). Zeigt den tatsächlichen Inhalt abgeschlossener LLM-Anfragen
(System-Prompt, Nachrichtenverlauf, generierte Antwort inkl. Denkprozess) -
siehe conversation_tracker.py für die Aufzeichnung selbst (nur main.py gen()/
ollama_compat.py api_chat() schreiben dort hinein, diese Seite hier ist reine
Lese-/Live-Anzeige + Löschen, analog zum Muster in cost_dashboard.py).

Tabelle + "Anzeigen"-Button-Modal exakt nach dem Muster von rag_dashboard.py
(dortiges Text-Modal), nur mit einem zweiten Tab für die rohe JSON-Ansicht
statt Rendered/Raw-Text, und der Bubble-/Markdown-Darstellung aus
chat_dashboard.py für die eigentliche Konversation (eigene Kopie hier, siehe
dortiger Kommentar zu "kein CDN/keine externe Bibliothek" - jede Dashboard-
Seite ist bewusst ein einziges, unabhängiges HTML-Dokument)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from . import conversation_tracker
from .config import get_config
from .dashboard import _LANGUAGES_JS, _is_disconnect_race

router = APIRouter()

HEARTBEAT_SECONDS = 5.0


async def build_snapshot() -> dict:
    return {"records": conversation_tracker.list_records()}


@router.get("/dashboard/conversations/status")
async def conversations_status():
    return await build_snapshot()


@router.websocket("/dashboard/conversations/ws")
async def conversations_ws(websocket: WebSocket):
    await websocket.accept()
    cfg = get_config()

    if cfg.api_key.enabled:
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        except Exception:
            await websocket.close(code=4401)
            return
        if not cfg.api_key.key or first.get("api_key") != cfg.api_key.key:
            try:
                await websocket.send_json({"type": "auth_error"})
            except Exception:
                pass
            await websocket.close(code=4401)
            return

    q = conversation_tracker.subscribe()
    try:
        await websocket.send_json(await build_snapshot())
        while True:
            try:
                await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                pass
            await websocket.send_json(await build_snapshot())
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if not _is_disconnect_race(e):
            raise
    finally:
        conversation_tracker.unsubscribe(q)


@router.get("/dashboard/conversations")
async def conversations_dashboard_page():
    return HTMLResponse(CONVERSATIONS_DASHBOARD_HTML)


CONVERSATIONS_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Hub – Conversations</title>
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.dataTables.min.css">
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.inputPaging.min.css">
<style>
  :root {
    --bg:#f5f6f8; --panel:#ffffff; --panel-2:#eef0f4; --border:#dfe3ea;
    --text:#161922; --text-dim:#4b5363; --mono: "SF Mono", Consolas, "Liberation Mono", monospace;
    --accent:#2563eb; --good:#15803d; --warn:#b45309; --bad:#dc2626;
    --accent-bg:rgba(37,99,235,.10); --good-bg:rgba(21,128,61,.10); --bad-bg:rgba(220,38,38,.10); --warn-bg:rgba(180,83,9,.12);
    --bubble-user:var(--accent); --bubble-user-text:#fff; --bubble-assistant:var(--panel-2);
    --code-bg:#0d1117; --code-text:#e6edf3; --code-bar:#161b22;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
      --text:#eef1f6; --text-dim:#a3acc2;
      --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
      --accent-bg:rgba(122,168,255,.15); --good-bg:rgba(74,222,128,.15); --bad-bg:rgba(248,113,113,.15); --warn-bg:rgba(251,191,36,.15);
      --bubble-user:var(--accent); --bubble-user-text:#0b0e14; --bubble-assistant:var(--panel-2);
    }
  }
  :root[data-theme="dark"] {
    --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
    --text:#eef1f6; --text-dim:#a3acc2;
    --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
    --accent-bg:rgba(122,168,255,.15); --good-bg:rgba(74,222,128,.15); --bad-bg:rgba(248,113,113,.15);
    --bubble-user:var(--accent); --bubble-user-text:#0b0e14; --bubble-assistant:var(--panel-2);
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; padding: 24px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  a { color: var(--accent); }
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom: 4px; }
  .topbar-actions { display:flex; gap:8px; align-items:flex-start; flex:0 0 auto; }
  .sub { color: var(--text-dim); font-size: 13px; margin-bottom: 20px; }
  .conn { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--text-dim); }
  .dot { width:8px; height:8px; border-radius:50%; background: var(--bad); }
  .dot.live { background: var(--good); box-shadow: 0 0 6px var(--good); }
  #theme-toggle, #lang-select, .btn {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; height:36px; padding:0 12px; font-size:13px; cursor:pointer;
  }
  #theme-toggle { width:36px; padding:0; font-size:16px; }
  #theme-toggle:hover, #lang-select:hover, .btn:hover { background:var(--panel-2); }
  .btn:disabled { opacity:.5; cursor:default; }
  .btn.danger:hover:not(:disabled) { border-color: var(--bad); color: var(--bad); }
  section { margin-bottom: 26px; }
  section h2 { font-size: 14px; color: var(--text-dim); text-transform:uppercase; letter-spacing:.04em; margin: 0 0 10px; }
  .banner { background:var(--accent-bg); color:var(--accent); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-size:13px; margin-bottom:20px; }
  table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .table-scroll { overflow-x:auto; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); vertical-align: middle; }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  td.mono, th.mono { font-family: var(--mono); }
  td.num, th.num { text-align:right; }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.ok { background: var(--good-bg); color: var(--good); }
  .badge.error { background: var(--bad-bg); color: var(--bad); }
  .badge.warn { background: var(--warn-bg); color: var(--warn); }
  .row-del { background:none; border:none; color:var(--text-dim); cursor:pointer; font-size:14px; padding:2px 6px; border-radius:6px; flex:0 0 auto; }
  .row-del:hover { color:var(--bad); background:var(--bad-bg); }
  .row-view { background:none; border:none; color:var(--accent); cursor:pointer; font-size:13px; padding:2px 4px; flex:0 0 auto; }
  .row-view:hover { text-decoration:underline; }
  /* Beide Buttons in EINER Zeile statt (bei knapper Spaltenbreite) untereinander
     umzubrechen - vorher standen sie als zwei separate Inline-Elemente mit
     einem bloßen Leerzeichen dazwischen im td, ein natürlicher Umbruchpunkt. */
  .row-actions { display:flex; align-items:center; gap:4px; white-space:nowrap; }
  .actions-row { display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  .actions-row .spacer { flex:1; }
  input[type=checkbox] { width:15px; height:15px; cursor:pointer; }
  .preview-cell { color:var(--text-dim); max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:inline-block; vertical-align:middle; }
  /* Kurze, feste Werte (Zeit/App/Tokens/Dauer/Status/Aktionen) sollen nie
     mehrzeilig umbrechen (sah vorher z.B. bei Zeit/App abgehackt aus) - die
     Tabelle darf dafür stattdessen horizontal scrollen (siehe .table-scroll),
     nur Modell/Preview dürfen wachsen. */
  #records-table td, #records-table th { white-space: nowrap; }
  .app-footer {
    margin-top:32px; padding-top:16px; border-top:1px solid var(--border);
    display:flex; align-items:center; justify-content:center; gap:6px;
    font-size:12px; color:var(--text-dim); flex-wrap:wrap;
  }
  .app-footer a { display:inline-flex; align-items:center; gap:4px; color:var(--text-dim); text-decoration:none; }
  .app-footer a:hover { color:var(--accent); }
  .app-footer img { width:16px; height:16px; border-radius:50%; object-fit:cover; flex:0 0 auto; }
  .app-footer .claude-mark { display:inline-flex; align-items:center; gap:4px; }
  .app-footer .sep { opacity:.5; }

  /* DataTables-Theming, identisch zu cost_dashboard.py/rag_dashboard.py - siehe dortigen Kommentar. */
  :root {
    --dt-control_color: var(--text-dim);
    --dt-body_border: 1px solid var(--border);
    --dt-header_border: 1px solid var(--border);
    --dt-footer_border: 1px solid var(--border);
    --dt-input_background: var(--panel);
    --dt-input_border: 1px solid var(--border);
    --dt-input_border-radius: 6px;
    --dt-input_color: var(--text);
    --dt-paging-button_background: var(--panel);
    --dt-paging-button_background-hover: var(--panel-2);
    --dt-paging-button_background-current: var(--accent);
    --dt-paging-button_background-current-hover: var(--accent);
    --dt-paging-button_background-disabled: transparent;
    --dt-paging-button_border: 1px solid var(--border);
    --dt-paging-button_border-hover: 1px solid var(--border);
    --dt-paging-button_border-current: 1px solid var(--accent);
    --dt-paging-button_border-current-hover: 1px solid var(--accent);
    --dt-paging-button_border-disabled: 1px solid transparent;
    --dt-paging-button_border-radius: 6px;
    --dt-paging-button_color: var(--text);
    --dt-paging-button_color-hover: var(--text);
    --dt-paging-button_color-current: #fff;
    --dt-paging-button_color-current-hover: #fff;
    --dt-paging-button_color-disabled: var(--text-dim);
  }
  .dt-container { color:var(--text); font-family:inherit; margin-top:10px; }
  .dt-container .dt-length select, .dt-paging-input input {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:6px; padding:4px 6px; font-size:13px;
  }
  .dt-paging-input input { width:3.5em; text-align:center; }
  table.dataTable tbody tr:hover td { background:var(--panel-2); }

  /* Modal: bewusst groß & breit (siehe Anfrage) - deutlich mehr Platz als das
     schmalere Text-Modal in rag_dashboard.py, damit eine ganze Konversation
     inkl. System-Prompt übersichtlich nebeneinander/untereinander Platz hat.
     Kopf (Titel/Meta/Params) + Tab-Umschalter bleiben als eigene Flex-Items
     fix stehen, nur .modal-body scrollt - bei den teils riesigen Konversationen
     (System-Prompts von >40k Zeichen sind keine Seltenheit, siehe Copilot)
     musste man vorher bis ganz nach oben zurückscrollen, um Tab zu wechseln
     oder das Modal zu schließen. */
  .modal-overlay {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.5);
    align-items:center; justify-content:center; z-index:100; padding:20px;
  }
  .modal-overlay.open { display:flex; }
  .modal {
    position:relative; background:var(--panel); border:1px solid var(--border); border-radius:12px;
    max-width:1200px; width:95vw; max-height:92vh; padding:24px 26px 0;
    display:flex; flex-direction:column; overflow:hidden;
  }
  .modal .close-btn {
    position:absolute; top:16px; right:20px; background:none; border:none;
    color:var(--text-dim); font-size:20px; cursor:pointer; line-height:1; z-index:1;
  }
  .modal-head { margin-bottom:14px; padding-right:30px; flex:0 0 auto; }
  .modal-head h3 { margin:0 0 8px; font-size:16px; }
  .modal-meta { display:flex; flex-wrap:wrap; gap:8px 18px; font-size:12.5px; color:var(--text-dim); }
  .modal-meta b { color:var(--text); font-weight:600; }
  .modal-params { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
  /* Werte werden in fmtParamValue() auf 60 Zeichen gekappt (voller Wert im
     title) - ohne diese Kappung riss z.B. ein "tools"-Param mit vielen
     Function-Definitionen (mehrere KB an JSON ohne ein einziges Leerzeichen,
     also ohne Umbruchstelle) das ganze Modal in der Breite auf. "tools"/
     "functions" selbst werden erst gar nicht als Chip gerendert, siehe
     #conv-modal-tools weiter unten. */
  .param-chip { background:var(--panel-2); border:1px solid var(--border); border-radius:20px; padding:2px 10px; font-size:11px; font-family:var(--mono); color:var(--text-dim); max-width:100%; overflow-wrap:anywhere; }

  .view-toggle { display:flex; gap:6px; margin:14px 0; flex:0 0 auto; }
  .view-toggle button {
    background:var(--panel); border:1px solid var(--border); color:var(--text-dim);
    border-radius:8px; height:30px; padding:0 12px; font-size:12px; cursor:pointer;
  }
  .view-toggle button.active { background:var(--accent); border-color:var(--accent); color:#fff; }

  .modal-body { flex:1 1 auto; overflow-y:auto; padding-bottom:24px; }

  /* System-Prompt: als <details> statt eines immer offenen, festen 220px-
     Kästchens - bei über 40k Zeichen (siehe Copilot-System-Prompt) war das
     nur ein winziges Scroll-Fenster voller kaum lesbarem Fließtext. Jetzt
     per Default eingeklappt (Zeichenzahl im Summary), Monospace statt
     Proportionalschrift (passt zu den <tag>-durchsetzten Prompts), eigener
     Copy-Button - und erst beim Aufklappen bis zu 400px hoch scrollbar. */
  .system-box {
    background:var(--panel-2); border:1px solid var(--border); border-radius:10px;
    margin-bottom:16px; overflow:hidden;
  }
  .system-box summary {
    cursor:pointer; list-style:none; user-select:none;
    display:flex; align-items:center; gap:10px; padding:10px 14px;
  }
  .system-box summary::-webkit-details-marker { display:none; }
  .system-box summary::before { content:"▶"; font-size:9px; color:var(--text-dim); transition:transform .12s; flex:0 0 auto; }
  .system-box[open] summary::before { transform:rotate(90deg); }
  .system-box-label { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--text-dim); font-weight:600; }
  .system-box-size { font-size:11.5px; color:var(--text-dim); }
  .system-box summary .mini-btn { margin-left:auto; }
  .system-box-text {
    margin:0; padding:0 14px 14px; font-family:var(--mono); font-size:12px;
    white-space:pre-wrap; word-break:break-word; max-height:400px; overflow-y:auto;
  }
  .mini-btn {
    background:var(--panel); border:1px solid var(--border); color:var(--text-dim);
    border-radius:5px; padding:2px 9px; font-size:11px; cursor:pointer;
  }
  .mini-btn:hover { border-color:var(--accent); color:var(--accent); }

  /* Tools-Liste: die vom Client mitgeschickten Function-Definitionen (bei
     Copilot z.B. 23 Stück) als Namens-Chips statt als ein einziger, riesiger
     JSON-Klumpen im Parameter-Bereich - siehe .param-chip-Kommentar oben. */
  .tools-box { background:var(--panel-2); border:1px solid var(--border); border-radius:10px; margin-top:10px; }
  .tools-box summary {
    cursor:pointer; list-style:none; user-select:none; padding:8px 14px;
    font-size:12px; color:var(--text-dim); display:flex; align-items:center; gap:6px;
  }
  .tools-box summary::-webkit-details-marker { display:none; }
  .tools-box summary::before { content:"▶"; font-size:9px; transition:transform .12s; }
  .tools-box[open] summary::before { transform:rotate(90deg); }
  .tools-list { padding:0 14px 12px; display:flex; flex-wrap:wrap; gap:6px; }
  .tools-list-item {
    background:var(--panel); border:1px solid var(--border); border-radius:6px;
    padding:3px 8px; font-size:11.5px; font-family:var(--mono); color:var(--text); cursor:help;
  }

  .raw-json-bar { display:flex; justify-content:flex-end; margin-bottom:8px; }
  .raw-json {
    background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
    padding:12px; font-family:var(--mono); font-size:12px; white-space:pre-wrap; word-break:break-word; margin:0;
  }

  /* Sehr lange Nachrichten (z.B. der 19KB-Workspace-Kontext, den Copilot vor
     die eigentliche Frage packt) sonst dominieren sie das ganze Modal - ab
     LONG_MSG_THRESHOLD Zeichen (siehe JS) auf ~220px geklammert, mit
     Verlaufs-Fade + "Mehr anzeigen"-Button zum Aufklappen. */
  .msg-collapsible { position:relative; max-height:220px; overflow:hidden; }
  .msg-collapsible.expanded { max-height:none; }
  .msg-collapsible:not(.expanded)::after {
    content:""; position:absolute; left:0; right:0; bottom:0; height:48px;
    background:linear-gradient(to bottom, transparent, var(--bubble-assistant));
  }
  .msg-row.user .msg-collapsible:not(.expanded)::after { background:linear-gradient(to bottom, transparent, var(--bubble-user)); }
  .msg-toggle {
    display:block; margin-top:8px; background:none; border:none; color:inherit;
    opacity:.85; text-decoration:underline; font-size:12px; cursor:pointer; padding:0;
  }

  /* Bubble-/Markdown-/Reasoning-/Code-Block-Darstellung, 1:1 aus chat_dashboard.py übernommen. */
  .msg-row { display:flex; margin-bottom:14px; }
  .msg-row.user { justify-content:flex-end; }
  .msg-row.assistant, .msg-row.other { justify-content:flex-start; }
  .bubble {
    max-width: 85%; border-radius:14px; padding:10px 14px; font-size:14px; line-height:1.55;
    overflow-wrap:anywhere;
  }
  .msg-row.user .bubble { background:var(--bubble-user); color:var(--bubble-user-text); border-bottom-right-radius:4px; white-space:pre-wrap; }
  .msg-row.assistant .bubble, .msg-row.other .bubble { background:var(--bubble-assistant); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .bubble-role { font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--text-dim); margin-bottom:4px; font-weight:600; }
  .msg-row.user .bubble-role { color:rgba(255,255,255,.75); }
  .bubble p { margin: 0 0 8px; }
  .bubble p:last-child { margin-bottom:0; }
  .bubble ul, .bubble ol { margin: 4px 0 8px; padding-left: 22px; }
  .bubble h3, .bubble h4, .bubble h5, .bubble h6 { margin: 10px 0 6px; }
  .bubble code { font-family: var(--mono); background:var(--panel-2); border-radius:4px; padding:1px 5px; font-size:.92em; }
  .msg-row.user .bubble code { background:rgba(255,255,255,.18); }

  .reasoning { margin-bottom:8px; }
  .reasoning summary {
    cursor:pointer; font-size:12px; color:var(--text-dim); user-select:none;
    display:flex; align-items:center; gap:6px; list-style:none;
  }
  .reasoning summary::-webkit-details-marker { display:none; }
  .reasoning summary::before { content:"▶"; font-size:9px; transition:transform .12s; }
  .reasoning[open] summary::before { transform:rotate(90deg); }
  .reasoning .reasoning-body {
    margin-top:6px; padding:8px 10px; border-left:2px solid var(--border);
    color:var(--text-dim); font-size:12.5px; white-space:pre-wrap;
  }

  .code-block { margin: 8px 0; border-radius:8px; overflow:hidden; border:1px solid var(--border); }
  .code-block-bar {
    background:var(--code-bar); color:#9da5b4; font-family: var(--mono); font-size:11px;
    padding:5px 10px; display:flex; align-items:center; justify-content:space-between;
  }
  .code-copy-btn { background:transparent; border:1px solid #30363d; color:#9da5b4; border-radius:5px; padding:2px 8px; font-size:11px; cursor:pointer; }
  .code-copy-btn:hover { background:#30363d; color:#fff; }
  .code-block pre { margin:0; background:var(--code-bg); color:var(--code-text); padding:12px; overflow-x:auto; }
  .code-block code { font-family: var(--mono); font-size:12.5px; background:none; padding:0; }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1 data-i18n="conversations.title">Conversations</h1>
      <div class="sub">
        <a href="/dashboard" data-i18n="nav.dashboardLink">← Dashboard</a>
        · <span class="conn"><span class="dot" id="conn-dot"></span><span id="conn-text" data-i18n="nav.connecting">connecting…</span></span>
      </div>
    </div>
    <div class="topbar-actions">
      <select id="lang-select" data-i18n-title="lang.selectTitle" title="Language"></select>
      <button id="theme-toggle" data-i18n-title="theme.toggleTitle" title="Toggle theme">🌙</button>
    </div>
  </div>

  <div class="banner" data-i18n="conversations.hint.disclaimer">
    Records the actual prompts and generated output of requests that reached a model - system prompt, message history, and response (incl. thinking). Delete anytime below.
  </div>

  <section>
    <h2 data-i18n="conversations.section.records">All Conversations</h2>
    <div class="actions-row">
      <button class="btn danger" id="delete-selected-btn" disabled>—</button>
      <button class="btn danger" id="reset-all-btn" data-i18n="conversations.action.resetAll">Reset all</button>
      <div class="spacer"></div>
      <span class="hint" id="records-status"></span>
    </div>
    <div class="table-scroll"><table id="records-table" class="display" style="width:100%"></table></div>
  </section>

  <footer class="app-footer">
    <span>© 2026</span>
    <a href="https://github.com/ridersonthecode" target="_blank" rel="noopener noreferrer" title="ridersonthecode on GitHub">
      <img src="https://github.com/ridersonthecode.png?s=64" alt="ridersonthecode" loading="lazy">
      ridersonthecode
    </a>
    <span class="sep">·</span>
    <span class="claude-mark" title="Entwickelt mit Claude Code">
      <img src="https://claude.ai/images/claude_app_icon.png" alt="Claude" loading="lazy">
      Entwickelt mit Claude Code
    </span>
  </footer>

  <div class="modal-overlay" id="conv-modal-overlay">
    <div class="modal">
      <button class="close-btn" id="conv-modal-close">✕</button>
      <div class="modal-head">
        <h3 id="conv-modal-title">–</h3>
        <div class="modal-meta" id="conv-modal-meta"></div>
        <div class="modal-params" id="conv-modal-params"></div>
      </div>
      <div class="view-toggle" id="conv-modal-toggle">
        <button id="conv-modal-view-chat" data-i18n="conversations.modal.tabConversation">Conversation</button>
        <button id="conv-modal-view-raw" data-i18n="conversations.modal.tabRaw">Raw JSON</button>
      </div>
      <div class="modal-body">
        <details id="conv-modal-system" class="system-box" style="display:none;">
          <summary>
            <span class="system-box-label" data-i18n="conversations.modal.systemPrompt">System Prompt</span>
            <span class="system-box-size" id="conv-modal-system-size"></span>
            <button type="button" class="mini-btn" id="conv-modal-system-copy" data-i18n="chat.action.copyCode">Copy</button>
          </summary>
          <pre id="conv-modal-system-text" class="system-box-text"></pre>
        </details>
        <details id="conv-modal-tools" class="tools-box" style="display:none;">
          <summary id="conv-modal-tools-label"></summary>
          <div class="tools-list" id="conv-modal-tools-list"></div>
        </details>
        <div id="conv-modal-chat"></div>
        <div class="raw-json-wrap" id="conv-modal-raw-wrap" style="display:none;">
          <div class="raw-json-bar"><button type="button" class="mini-btn" id="conv-modal-raw-copy" data-i18n="chat.action.copyCode">Copy</button></div>
          <pre class="raw-json" id="conv-modal-raw"></pre>
        </div>
      </div>
    </div>
  </div>

<script src="/static/vendor/datatables/dataTables.min.js"></script>
<script src="/static/vendor/datatables/dataTables.dataTables.min.js"></script>
<script src="/static/vendor/datatables/dataTables.inputPaging.min.js"></script>
<script>
const $ = (id) => document.getElementById(id);

// --- i18n (identisch zu den übrigen Dashboard-Seiten) ---------------------
const TRANSLATIONS = __TRANSLATIONS_JSON__;
const DEFAULT_LANG = "en";
const LANG_NAMES = { en: "English", de: "Deutsch" };
let currentLang = localStorage.getItem("vllm_dashboard_lang");
if (!currentLang || !TRANSLATIONS[currentLang]) currentLang = DEFAULT_LANG;

function t(key, vars) {
  const table = TRANSLATIONS[currentLang] || {};
  const fallback = TRANSLATIONS[DEFAULT_LANG] || {};
  let s = table[key] ?? fallback[key] ?? key;
  if (vars) for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
  return s;
}
function localeFor(lang) { return lang === "de" ? "de-DE" : "en-US"; }
function esc(s) { return (s ?? "").toString().replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

function populateLangSelect() {
  const sel = $("lang-select");
  sel.innerHTML = Object.keys(TRANSLATIONS).sort().map(code =>
    `<option value="${code}">${LANG_NAMES[code] || code}</option>`
  ).join("");
  sel.value = currentLang;
}
function applyStaticI18n() {
  document.documentElement.lang = currentLang;
  document.title = "LLM Hub – " + t("conversations.title");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  updateConnText();
  updateDeleteBtn();
}
$("lang-select").addEventListener("change", (e) => {
  currentLang = e.target.value;
  localStorage.setItem("vllm_dashboard_lang", currentLang);
  applyStaticI18n();
  initRecordsTable();
  if (latestSnapshot) render(latestSnapshot);
});
function updateConnText() { $("conn-text").textContent = t("nav." + connState); }

// --- Theme -----------------------------------------------------------------
function updateToggleIcon(theme) { $("theme-toggle").textContent = theme === "dark" ? "☀️" : "🌙"; }
function currentTheme() {
  return document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
(function initTheme() {
  const saved = localStorage.getItem("vllm_dashboard_theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  updateToggleIcon(currentTheme());
})();
$("theme-toggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("vllm_dashboard_theme", next);
  updateToggleIcon(next);
});

// --- API-Key (falls aktiviert, gleicher Key wie die übrigen Dashboard-Seiten) --
let apiKey = sessionStorage.getItem("vllm_dashboard_key") || "";
function authHeaders(extra) {
  const h = Object.assign({ "Content-Type": "application/json" }, extra || {});
  if (apiKey) h["Authorization"] = "Bearer " + apiKey;
  return h;
}

function copyText(text) {
  // Siehe chat_dashboard.py copyText() - navigator.clipboard nur in sicheren
  // Kontexten, dieses Dashboard läuft absichtlich auch über reines HTTP/LAN.
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  return new Promise((resolve, reject) => {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.top = "-1000px"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.focus(); ta.select();
    try { const ok = document.execCommand("copy"); document.body.removeChild(ta); ok ? resolve() : reject(new Error("copy failed")); }
    catch (e) { document.body.removeChild(ta); reject(e); }
  });
}

// --- Mini-Markdown-Renderer (1:1 aus chat_dashboard.py, siehe dortigen Kommentar) --
function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function renderInline(text) {
  // Platzhalter-Marker aus einem echten NUL-Zeichen, bewusst per
  // fromCharCode statt eines Escapes direkt im Quelltext erzeugt (kommt in
  // normalem Markdown/Code praktisch nie vor, kollidiert also nicht mit
  // echtem Text - anders als z.B. Leerzeichen).
  const NUL = String.fromCharCode(0);
  const codes = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => { codes.push(`<code>${escapeHtml(code)}</code>`); return NUL + (codes.length - 1) + NUL; });
  text = escapeHtml(text);
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  text = text.replace(new RegExp(NUL + "(\\d+)" + NUL, "g"), (_, i) => codes[Number(i)]);
  return text;
}
function renderMarkdownBlock(text) {
  const lines = text.split("\n");
  const htmlParts = [];
  let listBuf = [], listType = null, paraBuf = [];
  const flushList = () => {
    if (listBuf.length) {
      const tag = listType === "ol" ? "ol" : "ul";
      htmlParts.push(`<${tag}>${listBuf.map(li => `<li>${renderInline(li)}</li>`).join("")}</${tag}>`);
      listBuf = []; listType = null;
    }
  };
  const flushPara = () => { if (paraBuf.length) { htmlParts.push(`<p>${renderInline(paraBuf.join(" "))}</p>`); paraBuf = []; } };
  for (const line of lines) {
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    if (line.trim() === "") { flushPara(); flushList(); continue; }
    if (heading) { flushPara(); flushList(); const level = heading[1].length + 2; htmlParts.push(`<h${level}>${renderInline(heading[2])}</h${level}>`); }
    else if (ol) { flushPara(); if (listType !== "ol") flushList(); listType = "ol"; listBuf.push(ol[1]); }
    else if (ul) { flushPara(); if (listType !== "ul") flushList(); listType = "ul"; listBuf.push(ul[1]); }
    else { flushList(); paraBuf.push(line); }
  }
  flushPara(); flushList();
  return htmlParts.join("");
}
function renderCodeBlock(lang, code) {
  const id = "code-" + Math.random().toString(36).slice(2, 9);
  return `<div class="code-block">
    <div class="code-block-bar">
      <span>${esc(lang || "text")}</span>
      <button class="code-copy-btn" data-target="${id}">${esc(t("chat.action.copyCode"))}</button>
    </div>
    <pre><code id="${id}">${escapeHtml(code)}</code></pre>
  </div>`;
}
function renderMarkdown(text) {
  const parts = (text || "").split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) html += renderMarkdownBlock(parts[i]);
    else {
      const seg = parts[i];
      const nl = seg.indexOf("\n");
      const lang = nl === -1 ? seg.trim() : seg.slice(0, nl).trim();
      const code = nl === -1 ? "" : seg.slice(nl + 1);
      html += renderCodeBlock(lang, code);
    }
  }
  return html;
}
// Delegiert auf das Modal statt pro Code-Block einen eigenen Listener zu binden -
// das Modal wird bei jedem Öffnen komplett neu befüllt (innerHTML), siehe
// chat_dashboard.py für dasselbe Muster/denselben Grund.
$("conv-modal-chat").addEventListener("click", async (e) => {
  const btn = e.target.closest(".code-copy-btn");
  if (!btn) return;
  const codeEl = $(btn.dataset.target);
  if (!codeEl) return;
  try {
    await copyText(codeEl.textContent);
    const original = btn.textContent;
    btn.textContent = t("chat.action.copied");
    setTimeout(() => { btn.textContent = original; }, 1500);
  } catch (err) { /* kein Fallback mehr übrig - egal, nicht kritisch */ }
});
// "Mehr anzeigen"-Umschalter für sehr lange Nachrichten, siehe wrapCollapsible() -
// gleiches Delegations-Muster wie oben, aus demselben Grund.
$("conv-modal-chat").addEventListener("click", (e) => {
  const btn = e.target.closest(".msg-toggle");
  if (!btn) return;
  const target = $(btn.dataset.target);
  if (!target) return;
  const expand = !target.classList.contains("expanded");
  target.classList.toggle("expanded", expand);
  btn.textContent = expand ? btn.dataset.less : btn.dataset.more;
});

// Welche App den Request geschickt hat (siehe cost_dashboard.py appCell) - unverändert, lang gekürzt mit vollem Wert im Tooltip.
function appCell(userAgent) {
  if (!userAgent) return `<span class="hint">${esc(t("app.unknown"))}</span>`;
  const short = userAgent.length > 24 ? userAgent.slice(0, 23) + "…" : userAgent;
  return `<span title="${esc(userAgent)}">${esc(short)}</span>`;
}
function statusBadge(status) {
  const map = { ok: ["ok", "status.ok"], error: ["error", "status.error"], cancelled: ["warn", "status.cancelled"], aborted_loop: ["warn", "status.abortedLoop"] };
  const [cls, key] = map[status] || ["error", "status.error"];
  return `<span class="badge ${cls}">${esc(t(key))}</span>`;
}

// content kann bei Vision-Requests eine Liste aus Text-/Bild-Teilen statt
// eines simplen Strings sein (OpenAI-Format, siehe conversation_tracker.py
// _build_preview für dieselbe Behandlung server-seitig).
function messageText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.filter(p => p && p.type === "text").map(p => p.text || "").join(" ");
  return "";
}

// Ab dieser Rohtext-Länge wird eine Nachricht/Antwort per Default auf ~220px
// geklammert (siehe .msg-collapsible) - sonst dominiert z.B. der 19KB-
// Workspace-Kontext, den Copilot vor die eigentliche Frage packt, das ganze
// Modal und die eigentliche (oft viel kürzere) Nachricht geht darüber unter.
const LONG_MSG_THRESHOLD = 2000;
let collapseIdCounter = 0;
function wrapCollapsible(bodyHtml, rawLen) {
  if (rawLen <= LONG_MSG_THRESHOLD) return bodyHtml;
  const id = "msgc-" + (collapseIdCounter++);
  const more = esc(t("conversations.modal.showMore", { n: rawLen.toLocaleString(localeFor(currentLang)) }));
  const less = esc(t("conversations.modal.showLess"));
  return `<div class="msg-collapsible" id="${id}">${bodyHtml}</div>` +
    `<button type="button" class="msg-toggle" data-target="${id}" data-more="${more}" data-less="${less}">${more}</button>`;
}

function messageBubbleHtml(m) {
  const role = m.role;
  const rowCls = role === "user" ? "user" : (role === "assistant" ? "assistant" : "other");
  const text = messageText(m.content);
  let roleLabel = "";
  if (role === "tool") roleLabel = "🔧 " + esc(m.name ? m.name : t("conversations.role.tool"));
  else if (role !== "user" && role !== "assistant") roleLabel = esc(role);
  // Assistant-Nachrichten aus vergangenen Runden ohne content, aber mit
  // tool_calls (Function-Calling) - als kompakten Code-Block zeigen statt
  // eine leere Bubble.
  let body;
  if (!text && Array.isArray(m.tool_calls) && m.tool_calls.length) {
    body = renderCodeBlock("tool_calls", JSON.stringify(m.tool_calls, null, 2));
  } else {
    body = wrapCollapsible(renderMarkdown(text || `<span class="hint">${esc(t("conversations.modal.noContent"))}</span>`), text.length);
  }
  return `<div class="msg-row ${rowCls}"><div class="bubble">${roleLabel ? `<div class="bubble-role">${roleLabel}</div>` : ""}${body}</div></div>`;
}

function outputBubbleHtml(rec) {
  if (!rec.output_content && !rec.output_reasoning) return "";
  const reasoningHtml = rec.output_reasoning
    ? `<details class="reasoning"><summary>${esc(t("chat.label.reasoning"))}</summary><div class="reasoning-body">${escapeHtml(rec.output_reasoning)}</div></details>`
    : "";
  const body = wrapCollapsible(renderMarkdown(rec.output_content || ""), (rec.output_content || "").length);
  return `<div class="msg-row assistant"><div class="bubble">${reasoningHtml}${body}</div></div>`;
}

// Voller (unbeschnittener) Wert landet im title-Attribut des Chips, die
// Kurzform hier direkt darin - siehe .param-chip-Kommentar für den Grund
// (ein "tools"-Param mit vielen Function-Definitionen z.B. wäre sonst ein
// mehrere-KB-Klumpen JSON ohne Umbruchstelle).
function fmtParamValueFull(v) { return typeof v === "object" ? JSON.stringify(v) : String(v); }
function fmtParamValue(v) {
  const s = fmtParamValueFull(v);
  return s.length > 60 ? s.slice(0, 57) + "…" : s;
}

// Die Tool-/Function-Definitionen, die der Client mitschickt (request_params.
// tools bzw. das ältere .functions) - werden NICHT als Parameter-Chip
// gerendert (siehe fmtParamValue-Kommentar), sondern extra als Namensliste
// in #conv-modal-tools, aufklappbar mit der Beschreibung im Tooltip.
const TOOLS_PARAM_KEYS = ["tools", "functions"];
function extractToolNames(list) {
  if (!Array.isArray(list)) return [];
  return list.map(entry => {
    if (entry && entry.type === "function" && entry.function) {
      return { name: entry.function.name || "?", desc: entry.function.description || "" };
    }
    if (entry && typeof entry.name === "string") return { name: entry.name, desc: entry.description || "" };
    return { name: fmtParamValue(entry), desc: "" };
  });
}

// Grobe Größenangabe für die Meta-Zeile (Zeichen als Näherung für Bytes reicht
// hier - geht nur darum, vor dem Lesen ein Gefühl für den Umfang zu geben).
function fmtSize(chars) {
  if (chars < 1000) return chars + " B";
  if (chars < 1e6) return (chars / 1000).toFixed(1) + " KB";
  return (chars / 1e6).toFixed(1) + " MB";
}
function conversationSize(rec) {
  let msgCount = 0, totalChars = 0;
  if (Array.isArray(rec.messages)) {
    msgCount = rec.messages.length;
    for (const m of rec.messages) totalChars += messageText(m.content).length;
  } else if (rec.prompt) {
    const prompts = Array.isArray(rec.prompt) ? rec.prompt : [rec.prompt];
    msgCount = prompts.length;
    totalChars += prompts.join("").length;
  }
  totalChars += (rec.output_content || "").length + (rec.output_reasoning || "").length;
  return { msgCount, totalChars };
}

let modalHasSystemPrompt = false; // siehe setModalView() weiter unten
let modalHasTools = false;
function openConversationModal(rec) {
  $("conv-modal-title").textContent = rec.model || "–";
  const finishedDate = new Date(rec.finished_at * 1000).toLocaleString(localeFor(currentLang));
  const { msgCount, totalChars } = conversationSize(rec);
  const metaParts = [
    `<span><b>${esc(t("th.time"))}:</b> ${esc(finishedDate)}</span>`,
    `<span><b>${esc(t("th.endpoint"))}:</b> ${esc(rec.path || "–")}</span>`,
    `<span><b>${esc(t("th.duration"))}:</b> ${(rec.duration_ms / 1000).toFixed(2)}s</span>`,
    `<span><b>${esc(t("th.status"))}:</b> ${statusBadge(rec.status)}</span>`,
    `<span><b>${esc(t("th.promptTokens"))}:</b> ${rec.prompt_tokens ?? "–"}</span>`,
    `<span><b>${esc(t("th.complTokens"))}:</b> ${rec.completion_tokens ?? "–"}</span>`,
    `<span>${esc(t("conversations.modal.messages", { n: msgCount }))} · ${esc(fmtSize(totalChars))}</span>`,
  ];
  if (rec.finish_reason) metaParts.push(`<span><b>${esc(t("th.finishReason"))}:</b> ${esc(rec.finish_reason)}</span>`);
  $("conv-modal-meta").innerHTML = metaParts.join("");

  const params = rec.request_params || {};
  const toolsKey = TOOLS_PARAM_KEYS.find(k => Array.isArray(params[k]) && params[k].length);
  const paramKeys = Object.keys(params).filter(k => k !== toolsKey && params[k] !== undefined && params[k] !== null && params[k] !== false);
  $("conv-modal-params").innerHTML = paramKeys.map(k =>
    `<span class="param-chip" title="${esc(k)}=${esc(fmtParamValueFull(params[k]))}">${esc(k)}=${esc(fmtParamValue(params[k]))}</span>`
  ).join("");

  modalHasTools = !!toolsKey;
  if (toolsKey) {
    const tools = extractToolNames(params[toolsKey]);
    $("conv-modal-tools-label").textContent = "🔧 " + t("conversations.modal.toolsAvailable", { n: tools.length });
    $("conv-modal-tools-list").innerHTML = tools.map(tl => `<span class="tools-list-item" title="${esc(tl.desc)}">${esc(tl.name)}</span>`).join("");
  }
  $("conv-modal-tools").open = false;

  const messages = rec.messages || null;
  const systemMsg = messages ? messages.find(m => m.role === "system") : null;
  // modalHasSystemPrompt statt direkt hier den display-Wert zu setzen - siehe
  // setModalView() unten, das beim Tab-Wechsel dieselbe Sichtbarkeit erneut
  // anwenden muss. Vorher stand dort fälschlich eine Prüfung auf den
  // (unter Umständen vom vorigen Modal-Aufruf stehengebliebenen) Textinhalt
  // von #conv-modal-system-text - ein Datensatz ohne System-Prompt zeigte
  // dadurch den System-Prompt des zuletzt geöffneten Datensatzes an.
  modalHasSystemPrompt = !!systemMsg;
  const systemText = systemMsg ? messageText(systemMsg.content) : "";
  $("conv-modal-system-text").textContent = systemText;
  $("conv-modal-system-size").textContent = systemText ? t("conversations.modal.chars", { n: systemText.length.toLocaleString(localeFor(currentLang)) }) : "";
  $("conv-modal-system").open = false; // per Default eingeklappt, siehe .system-box-Kommentar im CSS

  let chatHtml;
  if (messages) {
    chatHtml = messages.filter(m => m.role !== "system").map(messageBubbleHtml).join("") + outputBubbleHtml(rec);
  } else {
    // Legacy /v1/completions - kein messages-Array, nur ein roher Prompt-String (oder eine Liste davon).
    const promptText = Array.isArray(rec.prompt) ? rec.prompt.join("\n\n") : (rec.prompt || "");
    const promptBody = wrapCollapsible(escapeHtml(promptText).replace(/\n/g, "<br>"), promptText.length);
    chatHtml = `<div class="msg-row user"><div class="bubble"><div class="bubble-role">${esc(t("conversations.modal.prompt"))}</div>${promptBody}</div></div>` + outputBubbleHtml(rec);
  }
  $("conv-modal-chat").innerHTML = chatHtml || `<div class="empty">${esc(t("conversations.modal.noContent"))}</div>`;
  $("conv-modal-raw").textContent = JSON.stringify(rec, null, 2);

  setModalView("chat");
  $("conv-modal-overlay").classList.add("open");
}
function closeConversationModal() { $("conv-modal-overlay").classList.remove("open"); }
$("conv-modal-close").addEventListener("click", closeConversationModal);
$("conv-modal-overlay").addEventListener("click", (e) => { if (e.target.id === "conv-modal-overlay") closeConversationModal(); });
// Copy-Buttons für System-Prompt und Raw-JSON - stehen jeweils in einem
// <summary>/Balken, e.stopPropagation() verhindert dass der Klick zusätzlich
// das umgebende <details> auf-/zuklappt.
async function copyWithFeedback(btn, text) {
  try {
    await copyText(text);
    const original = btn.textContent;
    btn.textContent = t("chat.action.copied");
    setTimeout(() => { btn.textContent = original; }, 1500);
  } catch (err) { /* kein Fallback mehr übrig - egal, nicht kritisch */ }
}
$("conv-modal-system-copy").addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  copyWithFeedback(e.currentTarget, $("conv-modal-system-text").textContent);
});
$("conv-modal-raw-copy").addEventListener("click", (e) => {
  copyWithFeedback(e.currentTarget, $("conv-modal-raw").textContent);
});
function setModalView(view) {
  $("conv-modal-view-chat").classList.toggle("active", view === "chat");
  $("conv-modal-view-raw").classList.toggle("active", view === "raw");
  $("conv-modal-system").style.display = (view === "chat" && modalHasSystemPrompt) ? "" : "none";
  $("conv-modal-tools").style.display = (view === "chat" && modalHasTools) ? "" : "none";
  $("conv-modal-chat").style.display = view === "chat" ? "" : "none";
  $("conv-modal-raw-wrap").style.display = view === "raw" ? "" : "none";
}
$("conv-modal-view-chat").addEventListener("click", () => setModalView("chat"));
$("conv-modal-view-raw").addEventListener("click", () => setModalView("raw"));

// --- State -------------------------------------------------------------
let latestSnapshot = null;
let selected = new Set();
let recordsById = new Map();

function updateDeleteBtn() {
  const btn = $("delete-selected-btn");
  btn.disabled = selected.size === 0;
  btn.textContent = t("conversations.action.deleteSelected", { n: selected.size });
}

// --- Tabelle (DataTables, siehe /static/vendor/datatables/README.md) -------
let recordsTable = null;
let recordsFingerprint = null;
let currentRecords = [];

function initRecordsTable() {
  if (recordsTable) { recordsTable.destroy(); $("records-table").innerHTML = ""; }
  recordsFingerprint = null;
  recordsTable = new DataTable("#records-table", {
    data: [],
    order: [[1, "desc"]],
    pageLength: 25,
    layout: { bottomEnd: "inputPaging" },
    language: { emptyTable: t("conversations.empty.noRecords") },
    columns: [
      {
        title: `<input type="checkbox" id="select-all-cb">`, data: null, orderable: false,
        render: (r) => `<input type="checkbox" class="row-cb" data-id="${esc(r.id)}" ${selected.has(r.id) ? "checked" : ""}>`,
      },
      { title: t("th.time"), data: null, render: (r, type) => type !== "display" ? r.finished_at : new Date(r.finished_at * 1000).toLocaleString(localeFor(currentLang)) },
      { title: t("th.model"), data: null, render: (r, type) => type === "display" ? esc(r.model) : (r.model || "") },
      { title: t("th.app"), data: null, orderable: false, render: (r) => appCell(r.user_agent) },
      { title: t("th.endpoint"), data: null, render: (r, type) => type === "display" ? esc(r.path || "–") : (r.path || "") },
      { title: t("th.preview"), data: null, orderable: false, render: (r, type) => type === "display" ? `<span class="preview-cell" title="${esc(r.preview || "")}">${esc(r.preview || "–")}</span>` : (r.preview || "") },
      {
        // Ein Tokens-Feld statt zwei Spalten (Prompt/Compl. getrennt) - gleiches
        // "X / Y"-Format wie die Engine-Tabelle im Haupt-Dashboard (siehe dortiges
        // th.tokensPromptGen), spart Spaltenbreite ohne Informationsverlust.
        title: t("th.tokensPromptGen"), data: null, orderable: false,
        render: (r) => (r.prompt_tokens ?? "–") + " / " + (r.completion_tokens ?? "–"),
      },
      { title: t("th.duration"), data: null, render: (r, type) => type !== "display" ? r.duration_ms : (r.duration_ms / 1000).toFixed(2) + "s" },
      {
        title: t("th.status"), data: null,
        render: (r, type) => type !== "display" ? r.status : statusBadge(r.status),
      },
      {
        title: "", data: null, orderable: false,
        render: (r) => `<div class="row-actions"><button class="row-view" data-id="${esc(r.id)}">${esc(t("action.select"))}</button><button class="row-del" data-id="${esc(r.id)}" title="${esc(t('action.delete'))}">🗑</button></div>`,
      },
    ],
  });

  $("select-all-cb").addEventListener("change", (e) => {
    if (e.target.checked) currentRecords.forEach(r => selected.add(r.id));
    else selected.clear();
    document.querySelectorAll("#records-table tbody .row-cb").forEach(cb => { cb.checked = selected.has(cb.dataset.id); });
    updateDeleteBtn();
  });

  recordsTable.on("draw", () => {
    document.querySelectorAll("#records-table tbody .row-cb").forEach(cb => {
      cb.addEventListener("change", (e) => {
        const id = e.target.dataset.id;
        if (e.target.checked) selected.add(id); else selected.delete(id);
        $("select-all-cb").checked = currentRecords.length > 0 && currentRecords.every(r => selected.has(r.id));
        updateDeleteBtn();
      });
    });
    document.querySelectorAll("#records-table tbody .row-del").forEach(btn => {
      btn.addEventListener("click", () => deleteOne(btn.dataset.id));
    });
    document.querySelectorAll("#records-table tbody .row-view").forEach(btn => {
      btn.addEventListener("click", () => {
        const rec = recordsById.get(btn.dataset.id);
        if (rec) openConversationModal(rec);
      });
    });
  });
}

function updateRecordsTable(records) {
  const stillThere = new Set(records.map(r => r.id));
  selected = new Set([...selected].filter(id => stillThere.has(id)));
  currentRecords = records;
  recordsById = new Map(records.map(r => [r.id, r]));

  $("select-all-cb").checked = records.length > 0 && records.every(r => selected.has(r.id));
  updateDeleteBtn();

  const fp = JSON.stringify(records.map(r => [r.id, r.status])); // reicht als Änderungs-Erkennung, ohne bei jedem Heartbeat riesige Payloads zu vergleichen
  if (fp === recordsFingerprint) return;
  recordsFingerprint = fp;
  recordsTable.clear();
  recordsTable.rows.add(records);
  recordsTable.draw(false);
}

function render(data) {
  latestSnapshot = data;
  updateRecordsTable(data.records || []);
}

// --- Aktionen --------------------------------------------------------------
async function deleteOne(id) {
  try {
    const res = await fetch(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE", headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
}

$("delete-selected-btn").addEventListener("click", async () => {
  if (selected.size === 0) return;
  if (!confirm(t("conversations.confirm.deleteSelected", { n: selected.size }))) return;
  try {
    const res = await fetch("/conversations/delete", { method: "POST", headers: authHeaders(), body: JSON.stringify({ ids: [...selected] }) });
    if (!res.ok) throw new Error(await res.text());
    selected.clear();
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
});

$("reset-all-btn").addEventListener("click", async () => {
  if (!confirm(t("conversations.confirm.resetAll"))) return;
  try {
    const res = await fetch("/conversations/reset", { method: "POST", headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    selected.clear();
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
});

// --- WebSocket-Verbindung (identisch zu cost_dashboard.py) ------------------
let connState = "connecting";
let ws = null;
let reconnectTimer = null;

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  if (document.hidden) return;
  reconnectTimer = setTimeout(connect, 1500);
}

function connect() {
  if (document.hidden) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/dashboard/conversations/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({ api_key: apiKey }));
    $("conn-dot").classList.add("live");
    connState = "live";
    updateConnText();
  };

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    if (data.type === "auth_error") {
      const key = prompt(t("auth.apiKeyPrompt"));
      if (key) { apiKey = key; sessionStorage.setItem("vllm_dashboard_key", key); }
      ws.close();
      return;
    }
    render(data);
  };

  ws.onclose = () => {
    $("conn-dot").classList.remove("live");
    connState = document.hidden ? "paused" : "reconnecting";
    updateConnText();
    scheduleReconnect();
  };
  ws.onerror = () => ws.close();
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearTimeout(reconnectTimer);
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  } else if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
    connect();
  }
});

populateLangSelect();
applyStaticI18n();
initRecordsTable();
connect();
</script>
</body>
</html>
"""

CONVERSATIONS_DASHBOARD_HTML = CONVERSATIONS_DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
