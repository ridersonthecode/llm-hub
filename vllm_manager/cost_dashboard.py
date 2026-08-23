"""Kostentracking-Seite: /dashboard/costs (HTML), /dashboard/costs/status
(JSON-Snapshot), /dashboard/costs/ws (WebSocket, Live-Push bei jeder
neuen/gelöschten Anfrage + Heartbeat). Rein fiktive Kostenkalkulation ggü.
einer Cloud-API (Default: Claude-Sonnet-5-Preise) - siehe cost_tracker.py.
Löschen/Reset laufen über main.py (/costs/*), diese Seite hier ist nur
Lese-/Live-Anzeige, analog zum Muster in dashboard.py."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from . import cost_tracker
from .config import get_config
from .dashboard import _LANGUAGES_JS

router = APIRouter()

HEARTBEAT_SECONDS = 5.0  # Kostendaten ändern sich nicht sekündlich wie GPU/RAM - reicht als Fallback


async def build_snapshot() -> dict:
    return {
        "summary": cost_tracker.summary(),
        "records": cost_tracker.list_records(),
    }


@router.get("/dashboard/costs/status")
async def cost_status():
    return await build_snapshot()


@router.websocket("/dashboard/costs/ws")
async def cost_ws(websocket: WebSocket):
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

    q = cost_tracker.subscribe()
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
    finally:
        cost_tracker.unsubscribe(q)


@router.get("/dashboard/costs")
async def cost_dashboard_page():
    return HTMLResponse(COST_DASHBOARD_HTML)


COST_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Manager – Costs</title>
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.dataTables.min.css">
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.inputPaging.min.css">
<style>
  :root {
    --bg:#f5f6f8; --panel:#ffffff; --panel-2:#eef0f4; --border:#dfe3ea;
    --text:#161922; --text-dim:#4b5363; --mono: "SF Mono", Consolas, "Liberation Mono", monospace;
    --accent:#2563eb; --good:#15803d; --warn:#b45309; --bad:#dc2626;
    --accent-bg:rgba(37,99,235,.10); --good-bg:rgba(21,128,61,.10); --bad-bg:rgba(220,38,38,.10); --warn-bg:rgba(180,83,9,.12);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
      --text:#eef1f6; --text-dim:#a3acc2;
      --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
      --accent-bg:rgba(122,168,255,.15); --good-bg:rgba(74,222,128,.15); --bad-bg:rgba(248,113,113,.15); --warn-bg:rgba(251,191,36,.15);
    }
  }
  :root[data-theme="dark"] {
    --bg:#0b0e14; --panel:#131722; --panel-2:#1a2030; --border:#2a3142;
    --text:#eef1f6; --text-dim:#a3acc2;
    --accent:#7aa8ff; --good:#4ade80; --warn:#fbbf24; --bad:#f87171;
    --accent-bg:rgba(122,168,255,.15); --good-bg:rgba(74,222,128,.15); --bad-bg:rgba(248,113,113,.15);
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
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:14px; margin-bottom: 20px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card .label { color:var(--text-dim); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
  .card .value { font-size:22px; font-weight:600; }
  .card .hint { color:var(--text-dim); font-size:12px; margin-top:6px; }
  table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .table-scroll { overflow-x:auto; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); vertical-align: middle; }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  td.mono, th.mono { font-family: var(--mono); }
  td.num, th.num { text-align:right; }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .banner { background:var(--accent-bg); color:var(--accent); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-size:13px; margin-bottom:20px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.ok { background: var(--good-bg); color: var(--good); }
  .badge.error { background: var(--bad-bg); color: var(--bad); }
  .row-del { background:none; border:none; color:var(--text-dim); cursor:pointer; font-size:14px; padding:2px 6px; border-radius:6px; }
  .row-del:hover { color:var(--bad); background:var(--bad-bg); }
  .actions-row { display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  .actions-row .spacer { flex:1; }
  input[type=checkbox] { width:15px; height:15px; cursor:pointer; }
  .unpriced { color:var(--text-dim); }
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

  /* DataTables: an das App-Theme anpassen (Bibliothek unter /static/vendor/datatables/, siehe README dort).
     Farben laufen wo moeglich ueber die von DataTables selbst vorgesehenen
     --dt-*-Variablen (siehe dataTables.dataTables.css) statt eigener Selektor-
     Overrides - deren Original-Regeln haben oft hoehere Spezifitaet als ein
     einfacher Klassen-Override, egal in welcher Reihenfolge im Dokument. */
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
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1 data-i18n="cost.title">Cost Tracking</h1>
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

  <div class="banner" data-i18n="cost.disclaimer">
    Purely fictional cost comparison: the local model runs for free - this estimates what the same tokens would have cost via the Claude Sonnet 5 API, for reference. Adjust rates per model in the Config Editor.
  </div>

  <section>
    <h2 data-i18n="cost.section.summary">Summary</h2>
    <div class="grid">
      <div class="card">
        <div class="label" data-i18n="cost.card.totalCost">Total Cost (fictional)</div>
        <div class="value" id="total-cost">–</div>
        <div class="hint" id="total-cost-hint"></div>
      </div>
      <div class="card">
        <div class="label" data-i18n="cost.card.totalRequests">Total Requests</div>
        <div class="value" id="total-requests">–</div>
      </div>
      <div class="card">
        <div class="label" data-i18n="cost.card.pricedRequests">Priced Requests</div>
        <div class="value" id="priced-requests">–</div>
        <div class="hint" data-i18n="cost.hint.pricedRequests">Requests without exact token usage show no cost</div>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cost.section.byModel">Cost by Model</h2>
    <div id="by-model-box"></div>
  </section>

  <section>
    <h2 data-i18n="cost.section.records">All Requests</h2>
    <div class="actions-row">
      <button class="btn danger" id="delete-selected-btn" disabled>—</button>
      <button class="btn danger" id="reset-all-btn" data-i18n="cost.action.resetAll">Reset all</button>
      <div class="spacer"></div>
      <span class="hint" id="records-status"></span>
    </div>
    <table id="records-table" class="display" style="width:100%"></table>
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

<script src="/static/vendor/datatables/dataTables.min.js"></script>
<script src="/static/vendor/datatables/dataTables.dataTables.min.js"></script>
<script src="/static/vendor/datatables/dataTables.inputPaging.min.js"></script>
<script>
const $ = (id) => document.getElementById(id);

// Siehe dashboard.py (gleiches Problem/gleiche Lösung): ersetzt das DOM nur
// bei tatsächlicher Änderung und schiebt ein Update auf, solange der Nutzer
// gerade Text innerhalb der Box markiert hat - der nächste Tick holt es nach.
function safeSetHTML(el, html) {
  if (!el) return false;
  if (el._lastHtml === html) return false;
  const sel = window.getSelection();
  if (sel && sel.rangeCount > 0 && !sel.isCollapsed && el.contains(sel.anchorNode)) {
    return false;
  }
  el.innerHTML = html;
  el._lastHtml = html;
  return true;
}

// --- i18n (identisch zum Haupt-Dashboard) ---------------------------------
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
function esc(s) { return (s ?? "").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function populateLangSelect() {
  const sel = $("lang-select");
  sel.innerHTML = Object.keys(TRANSLATIONS).sort().map(code =>
    `<option value="${code}">${LANG_NAMES[code] || code}</option>`
  ).join("");
  sel.value = currentLang;
}
function applyStaticI18n() {
  document.documentElement.lang = currentLang;
  document.title = "vLLM Manager – " + t("cost.title");
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

// --- API-Key (falls aktiviert, gleicher Key wie Haupt-Dashboard) -----------
let apiKey = sessionStorage.getItem("vllm_dashboard_key") || "";
function authHeaders(extra) {
  const h = Object.assign({ "Content-Type": "application/json" }, extra || {});
  if (apiKey) h["Authorization"] = "Bearer " + apiKey;
  return h;
}

function fmtUsd(x) {
  if (x === null || x === undefined) return null;
  if (x === 0) return "$0.00";
  if (x < 0.01) return "$" + x.toFixed(6);
  return "$" + x.toFixed(4);
}

// Welche App den Request geschickt hat, aus dem rohen User-Agent-Header
// (siehe telemetry.start_request) - unverändert angezeigt, lang truncated
// mit vollem Wert im Tooltip.
function appCell(userAgent) {
  if (!userAgent) return `<span class="unpriced">${esc(t("app.unknown"))}</span>`;
  const short = userAgent.length > 28 ? userAgent.slice(0, 27) + "…" : userAgent;
  return `<span title="${esc(userAgent)}">${esc(short)}</span>`;
}

// --- State -------------------------------------------------------------
let latestSnapshot = null;
let selected = new Set();

function updateDeleteBtn() {
  const btn = $("delete-selected-btn");
  btn.disabled = selected.size === 0;
  btn.textContent = t("cost.action.deleteSelected", { n: selected.size });
}

// --- All Requests (DataTables, siehe /static/vendor/datatables/README.md) --
// costs.jsonl wächst dauerhaft (nie automatisch aufgeräumt, siehe
// cost_tracker.py) - daher paginiert statt als eine lange Liste. Spaltentitel
// hängen von der Sprache ab -> bei Sprachwechsel komplett neu aufgebaut (siehe
// lang-select-Handler unten).
let recordsTable = null;
let recordsFingerprint = null;
let currentRecords = [];

function initRecordsTable() {
  if (recordsTable) { recordsTable.destroy(); $("records-table").innerHTML = ""; }
  recordsFingerprint = null;
  recordsTable = new DataTable("#records-table", {
    data: [],
    order: [],
    pageLength: 25,
    layout: { bottomEnd: "inputPaging" },
    language: { emptyTable: t("cost.empty.noRecords") },
    columns: [
      {
        title: `<input type="checkbox" id="select-all-cb">`, data: null, orderable: false,
        render: (r) => `<input type="checkbox" class="row-cb" data-id="${esc(r.id)}" ${selected.has(r.id) ? "checked" : ""}>`,
      },
      { title: t("th.time"), data: null, render: (r, type) => type !== "display" ? r.finished_at : new Date(r.finished_at * 1000).toLocaleString(localeFor(currentLang)) },
      { title: t("th.model"), data: null, render: (r, type) => type === "display" ? esc(r.model) : (r.model || "") },
      { title: t("th.app"), data: null, orderable: false, render: (r) => appCell(r.user_agent) },
      { title: t("th.endpoint"), data: null, render: (r, type) => type === "display" ? esc(r.path || "–") : (r.path || "") },
      { title: t("th.promptTokens"), data: null, render: (r, type) => type !== "display" ? (r.prompt_tokens ?? -1) : (r.prompt_tokens ?? "–") },
      { title: t("th.complTokens"), data: null, render: (r, type) => type !== "display" ? (r.completion_tokens ?? -1) : (r.completion_tokens ?? "–") },
      {
        title: t("th.cost"), data: null,
        render: (r, type) => {
          if (type !== "display") return r.cost_usd ?? -1;
          const cost = fmtUsd(r.cost_usd);
          return cost !== null
            ? `<span title="${esc(t("cost.hint.rateTooltip", { in: r.pricing_input_per_mtok, out: r.pricing_output_per_mtok }))}">${cost}</span>`
            : `<span class="unpriced" title="${esc(t("cost.hint.noUsage"))}">–</span>`;
        },
      },
      {
        title: t("th.status"), data: null,
        render: (r, type) => type !== "display" ? r.status : `<span class="badge ${r.status === 'ok' ? 'ok' : 'error'}">${r.status === 'ok' ? t("status.ok") : t("status.error")}</span>`,
      },
      { title: "", data: null, orderable: false, render: (r) => `<button class="row-del" data-id="${esc(r.id)}" title="${esc(t('action.delete'))}">🗑</button>` },
    ],
  });

  // Select-all: EIN Header-Element, bleibt über draw()-Aufrufe hinweg erhalten
  // (nur tbody wird pro draw() neu erzeugt) - Listener daher nur einmal hier
  // binden. Wirkt bewusst auf ALLE Datensätze, nicht nur die sichtbare Seite -
  // entspricht dem bisherigen Verhalten vor der Pagination.
  $("select-all-cb").addEventListener("change", (e) => {
    if (e.target.checked) currentRecords.forEach(r => selected.add(r.id));
    else selected.clear();
    document.querySelectorAll("#records-table tbody .row-cb").forEach(cb => { cb.checked = selected.has(cb.dataset.id); });
    updateDeleteBtn();
  });

  // Zeilen-Checkboxen/Löschen-Buttons werden bei jedem draw() (auch beim
  // Seitenwechsel) neu erzeugt - Listener daher hier statt einmalig binden.
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
  });
}

function updateRecordsTable(records) {
  // Auswahl bereinigen: Datensätze, die nicht mehr existieren (gelöscht/reset), rausnehmen.
  const stillThere = new Set(records.map(r => r.id));
  selected = new Set([...selected].filter(id => stillThere.has(id)));
  currentRecords = records;

  $("select-all-cb").checked = records.length > 0 && records.every(r => selected.has(r.id));
  updateDeleteBtn();

  const fp = JSON.stringify(records);
  if (fp === recordsFingerprint) return;
  recordsFingerprint = fp;
  recordsTable.clear();
  recordsTable.rows.add(records);
  recordsTable.draw(false);
}

function render(data) {
  latestSnapshot = data;
  const summary = data.summary || {};
  const records = data.records || [];

  $("total-cost").textContent = fmtUsd(summary.total_cost_usd) ?? "$0.00";
  $("total-cost-hint").textContent = t("cost.hint.vsClaudeSonnet5");
  $("total-requests").textContent = summary.total_requests ?? 0;
  $("priced-requests").textContent = (summary.priced_requests ?? 0) + " / " + (summary.total_requests ?? 0);

  const byModel = summary.by_model || [];
  if (byModel.length === 0) {
    safeSetHTML($("by-model-box"), `<div class="empty">${t("cost.empty.noRecords")}</div>`);
  } else {
    safeSetHTML($("by-model-box"), `<div class="table-scroll"><table><thead><tr>
      <th>${t("th.model")}</th><th class="num">${t("th.requests")}</th><th class="num">${t("cost.card.pricedRequests")}</th><th class="num">${t("th.cost")}</th>
      </tr></thead><tbody>` + byModel.map(m => `<tr>
        <td>${esc(m.model)}</td>
        <td class="mono num">${m.requests}</td>
        <td class="mono num">${m.priced_requests}</td>
        <td class="mono num">${fmtUsd(m.cost_usd) ?? "$0.00"}</td>
      </tr>`).join("") + `</tbody></table></div>`);
  }

  updateRecordsTable(records);
}

// --- Aktionen --------------------------------------------------------------
async function deleteOne(id) {
  try {
    const res = await fetch(`/costs/${encodeURIComponent(id)}`, { method: "DELETE", headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    // Nächster Live-Push (WS) aktualisiert die Tabelle automatisch.
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
}

$("delete-selected-btn").addEventListener("click", async () => {
  if (selected.size === 0) return;
  if (!confirm(t("cost.confirm.deleteSelected", { n: selected.size }))) return;
  try {
    const res = await fetch("/costs/delete", { method: "POST", headers: authHeaders(), body: JSON.stringify({ ids: [...selected] }) });
    if (!res.ok) throw new Error(await res.text());
    selected.clear();
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
});

$("reset-all-btn").addEventListener("click", async () => {
  if (!confirm(t("cost.confirm.resetAll"))) return;
  try {
    const res = await fetch("/costs/reset", { method: "POST", headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    selected.clear();
  } catch (e) {
    $("records-status").textContent = t("error.generic", { msg: e.message });
  }
});

// --- WebSocket-Verbindung ----------------------------------------------
let connState = "connecting";

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/dashboard/costs/ws");

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
    connState = "reconnecting";
    updateConnText();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

populateLangSelect();
applyStaticI18n();
initRecordsTable();
connect();
</script>
</body>
</html>
"""

COST_DASHBOARD_HTML = COST_DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
