"""RAG-Verwaltungsseite: /dashboard/rag (HTML). Eigene, einfache Seite (kein
WebSocket-Live-Push wie beim Haupt-Dashboard nötig - Dokumente ändern sich
nicht sekündlich, ein Neuladen per REST bei jeder Aktion reicht). Nutzt
dieselben Übersetzungen wie das Haupt-Dashboard (siehe dashboard.py)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .dashboard import _LANGUAGES_JS

router = APIRouter()


@router.get("/dashboard/rag")
async def rag_dashboard_page():
    return HTMLResponse(RAG_DASHBOARD_HTML)


RAG_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Manager – RAG</title>
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
  #theme-toggle, #lang-select, .btn, .tab-btn {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; height:36px; padding:0 12px; font-size:13px; cursor:pointer;
  }
  #theme-toggle { width:36px; padding:0; font-size:16px; }
  #theme-toggle:hover, #lang-select:hover, .btn:hover { background:var(--panel-2); }
  .btn:disabled { opacity:.5; cursor:default; }
  .btn.danger:hover { border-color: var(--bad); color: var(--bad); }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  section { margin-bottom: 26px; }
  section h2 { font-size: 14px; color: var(--text-dim); text-transform:uppercase; letter-spacing:.04em; margin: 0 0 10px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); vertical-align: top; }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  tr.clickable { cursor:pointer; }
  tr.clickable:hover { background: var(--panel-2); }
  tr.selected { background: var(--accent-bg); }
  td.mono, th.mono { font-family: var(--mono); }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .banner { background:var(--warn-bg); color:var(--warn); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-size:13px; margin-bottom:20px; }
  label { display:block; font-size:12px; color:var(--text-dim); margin: 12px 0 4px; }
  label:first-child { margin-top: 0; }
  input[type=text], input[type=number], textarea, select {
    width:100%; background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:8px 10px; font-size:13px; font-family: inherit;
  }
  textarea { min-height: 120px; resize: vertical; font-family: var(--mono); }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .row > div { flex: 1; min-width: 200px; }
  .tabs { display:flex; gap:6px; margin-bottom:12px; }
  .tab-btn.active { background: var(--accent); border-color: var(--accent); color:#fff; }
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }
  .actions-row { display:flex; justify-content:flex-end; margin-top:14px; gap:8px; align-items:center; }
  .status-msg { font-size:12px; color:var(--text-dim); }
  .status-msg.error { color: var(--bad); }
  .score-bar-bg { background:var(--panel-2); border-radius:6px; height:6px; overflow:hidden; margin-top:4px; width:80px; }
  .score-bar-fg { background:var(--accent); height:100%; }
  .snippet { color:var(--text-dim); font-size:12px; margin-top:4px; max-width:480px; cursor:pointer; }
  .snippet:hover { color:var(--text); text-decoration:underline; }
  .view-doc-btn { background:none; border:none; color:var(--accent); cursor:pointer; font-size:13px; padding:0; margin-right:10px; }
  .view-doc-btn:hover { text-decoration:underline; }

  .modal-overlay {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.5);
    align-items:center; justify-content:center; z-index:100; padding:20px;
  }
  .modal-overlay.open { display:flex; }
  .modal {
    position:relative; background:var(--panel); border:1px solid var(--border); border-radius:12px;
    max-width:720px; width:100%; max-height:85vh; overflow-y:auto; padding:22px;
  }
  .modal h3 { margin:0 0 4px; font-size:16px; word-break:break-all; }
  .modal .hint { margin:0 0 16px; }
  .modal .close-btn {
    position:absolute; top:16px; right:20px; background:none; border:none;
    color:var(--text-dim); font-size:20px; cursor:pointer; line-height:1;
  }
  .modal .chunk-block { margin-bottom:16px; }
  .modal .chunk-block:last-child { margin-bottom:0; }
  .modal .chunk-label { font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px; }
  .modal .chunk-text {
    background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
    padding:12px; font-family:var(--mono); font-size:12px; white-space:pre-wrap; word-break:break-word; margin:0;
  }
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
      <h1 data-i18n="rag.title">RAG Documents</h1>
      <div class="sub"><a href="/dashboard" data-i18n="nav.dashboardLink">← Dashboard</a></div>
    </div>
    <div class="topbar-actions">
      <select id="lang-select" data-i18n-title="lang.selectTitle" title="Language"></select>
      <button id="theme-toggle" data-i18n-title="theme.toggleTitle" title="Toggle theme">🌙</button>
    </div>
  </div>

  <div id="not-configured-banner" class="banner" style="display:none;" data-i18n="rag.notConfigured"></div>

  <div id="rag-content">
    <section>
      <h2 data-i18n="section.collections">Collections</h2>
      <div id="collections-box"></div>
    </section>

    <section>
      <h2 data-i18n="section.addContent">Add Content</h2>
      <div class="card">
        <div class="tabs">
          <button class="tab-btn active" id="tab-btn-text" data-i18n="form.textTab">Text</button>
          <button class="tab-btn" id="tab-btn-file" data-i18n="form.fileTab">File</button>
        </div>

        <div class="tab-panel active" id="tab-panel-text">
          <label data-i18n="form.textLabel">Text</label>
          <textarea id="add-text-content" data-i18n-placeholder="form.textPlaceholder" placeholder="Paste or type text…"></textarea>
          <label data-i18n="form.sourceLabel">Source label (optional)</label>
          <input type="text" id="add-text-source">
        </div>
        <div class="tab-panel" id="tab-panel-file">
          <label data-i18n="form.pathLabel">File path (on the server, PDF/TXT/MD)</label>
          <input type="text" id="add-file-path" placeholder="/home/user/docs/invoice.pdf">
        </div>

        <div class="row">
          <div>
            <label data-i18n="form.collectionLabel">Collection</label>
            <select id="add-collection-select"></select>
          </div>
          <div>
            <label>&nbsp;</label>
            <input type="text" id="add-collection-new" data-i18n-placeholder="form.newCollectionPlaceholder" placeholder="or type a new collection name">
          </div>
        </div>

        <div class="actions-row">
          <span class="status-msg" id="add-status"></span>
          <button class="btn primary" id="add-submit-btn" data-i18n="action.add">Add</button>
        </div>
      </div>
    </section>

    <section>
      <h2 data-i18n="section.documents">Documents</h2>
      <div id="documents-empty-hint" class="empty" data-i18n="empty.selectCollection"></div>
      <table id="documents-table" class="display" style="width:100%; display:none;"></table>
    </section>

    <section>
      <h2 data-i18n="section.testSearch">Test Search</h2>
      <div class="card">
        <div class="row">
          <div>
            <label data-i18n="form.collectionLabel">Collection</label>
            <select id="search-collection-select"></select>
          </div>
          <div style="max-width:120px;">
            <label data-i18n="form.topKLabel">Results</label>
            <input type="number" id="search-top-k" value="5" min="1" max="50">
          </div>
        </div>
        <label data-i18n="form.queryLabel">Query</label>
        <input type="text" id="search-query">
        <div class="actions-row">
          <span class="status-msg" id="search-status"></span>
          <button class="btn primary" id="search-submit-btn" data-i18n="action.search">Search</button>
        </div>
      </div>
      <div id="search-results-box" style="margin-top:14px;"></div>
    </section>
  </div>

  <div class="modal-overlay" id="text-modal-overlay">
    <div class="modal">
      <button class="close-btn" id="text-modal-close">✕</button>
      <h3 id="text-modal-title">–</h3>
      <p class="hint" id="text-modal-hint"></p>
      <div id="text-modal-body"></div>
    </div>
  </div>

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
  document.title = "vLLM Manager – " + t("rag.title");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
}
$("lang-select").addEventListener("change", (e) => {
  currentLang = e.target.value;
  localStorage.setItem("vllm_dashboard_lang", currentLang);
  applyStaticI18n();
  initDocumentsTable();
  refreshAll();
});

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
const apiKey = sessionStorage.getItem("vllm_dashboard_key") || "";
function authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  if (apiKey) h["Authorization"] = "Bearer " + apiKey;
  return h;
}

// --- Text-Detail-Modal ------------------------------------------------
// Zeigt gespeicherte Chunks vollständig (unabgeschnitten) an - für ein
// ganzes Dokument (mehrere Chunks, siehe openDocumentModal) oder einen
// einzelnen Treffer aus der Testsuche (siehe renderSearchResults).
function openTextModal(title, hint, chunks) {
  $("text-modal-title").textContent = title || "–";
  $("text-modal-hint").textContent = hint || "";
  $("text-modal-hint").style.display = hint ? "" : "none";
  $("text-modal-body").innerHTML = chunks.map(c => `
    <div class="chunk-block">
      ${chunks.length > 1 ? `<div class="chunk-label">${esc(t("modal.chunkLabel", { i: (c.chunk_index ?? 0) + 1, n: c.chunk_count ?? chunks.length }))}</div>` : ""}
      <pre class="chunk-text">${esc(c.text || "")}</pre>
    </div>`).join("");
  $("text-modal-overlay").classList.add("open");
}
function closeTextModal() { $("text-modal-overlay").classList.remove("open"); }
$("text-modal-close").addEventListener("click", closeTextModal);
$("text-modal-overlay").addEventListener("click", (e) => { if (e.target.id === "text-modal-overlay") closeTextModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeTextModal(); });

async function openDocumentModal(collection, documentId, title) {
  const res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/documents/${encodeURIComponent(documentId)}/chunks`, { headers: authHeaders() });
  if (!res.ok) { alert(t("error.generic", { msg: await res.text() })); return; }
  const data = await res.json();
  const chunks = data.chunks || [];
  const hint = chunks.length > 1 ? t("modal.chunkCountHint", { n: chunks.length }) : "";
  openTextModal(title, hint, chunks);
}

// --- Tabs --------------------------------------------------------------
$("tab-btn-text").addEventListener("click", () => switchTab("text"));
$("tab-btn-file").addEventListener("click", () => switchTab("file"));
function switchTab(name) {
  $("tab-btn-text").classList.toggle("active", name === "text");
  $("tab-btn-file").classList.toggle("active", name === "file");
  $("tab-panel-text").classList.toggle("active", name === "text");
  $("tab-panel-file").classList.toggle("active", name === "file");
}

// --- State -------------------------------------------------------------
let collectionsCache = [];
let selectedCollection = null;

function populateCollectionSelects() {
  const names = collectionsCache.map(c => c.name);
  for (const id of ["add-collection-select", "search-collection-select"]) {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    if (names.includes(prev)) sel.value = prev;
  }
}

async function loadCollections() {
  const res = await fetch("/rag/collections", { headers: authHeaders() });
  if (res.status === 400) {
    $("not-configured-banner").style.display = "block";
    $("rag-content").style.display = "none";
    return;
  }
  $("not-configured-banner").style.display = "none";
  $("rag-content").style.display = "block";
  const data = await res.json();
  collectionsCache = data.collections || [];
  renderCollections();
  populateCollectionSelects();
}

function renderCollections() {
  if (collectionsCache.length === 0) {
    $("collections-box").innerHTML = `<div class="empty">${t("empty.noCollections")}</div>`;
    return;
  }
  $("collections-box").innerHTML = `<table><thead><tr>
    <th>${t("th.collection")}</th><th>${t("th.chunks")}</th><th>${t("th.action")}</th>
    </tr></thead><tbody>` + collectionsCache.map(c => `
      <tr class="clickable ${c.name === selectedCollection ? 'selected' : ''}" data-name="${esc(c.name)}">
        <td>${esc(c.name)}</td>
        <td class="mono">${c.points_count}</td>
        <td><button class="btn danger del-collection-btn" data-name="${esc(c.name)}">${t("action.delete")}</button></td>
      </tr>`).join("") + `</tbody></table>`;

  $("collections-box").querySelectorAll("tr.clickable").forEach(tr => {
    tr.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      selectedCollection = tr.dataset.name;
      renderCollections();
      loadDocuments(selectedCollection);
    });
  });
  $("collections-box").querySelectorAll(".del-collection-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm(t("confirm.deleteCollection", { name: btn.dataset.name }))) return;
      await fetch(`/rag/collections/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE", headers: authHeaders() });
      if (selectedCollection === btn.dataset.name) {
        selectedCollection = null;
        documentsTable.table().container().style.display = "none";
        $("documents-empty-hint").style.display = "";
      }
      loadCollections();
    });
  });
}

// --- Documents (DataTables, siehe /static/vendor/datatables/README.md) -----
// Kann mit vielen Dokumenten pro Collection wachsen - daher paginiert.
// Spaltentitel hängen von der Sprache ab -> bei Sprachwechsel komplett neu
// aufgebaut (siehe lang-select-Handler unten).
let documentsTable = null;

function initDocumentsTable() {
  if (documentsTable) { documentsTable.destroy(); $("documents-table").innerHTML = ""; }
  documentsTable = new DataTable("#documents-table", {
    data: [],
    order: [],
    pageLength: 10,
    layout: { bottomEnd: "inputPaging" },
    language: { emptyTable: t("empty.noDocuments") },
    columns: [
      { title: t("th.source"), data: null, render: (d, type) => type === "display" ? esc(d.filename || d.source || "–") : (d.filename || d.source || "") },
      { title: t("th.chunks"), data: null, render: (d, type) => type !== "display" ? (d.chunk_count ?? -1) : (d.chunk_count ?? "–") },
      { title: t("th.addedAt"), data: null, render: (d, type) => type !== "display" ? (d.added_at ?? 0) : (d.added_at ? new Date(d.added_at * 1000).toLocaleString(localeFor(currentLang)) : "–") },
      {
        title: t("th.action"), data: null, orderable: false,
        render: (d) => `<button class="view-doc-btn" data-id="${esc(d.document_id)}" data-title="${esc(d.filename || d.source || "–")}">${t("action.select")}</button>`
          + `<button class="btn danger del-doc-btn" data-id="${esc(d.document_id)}">${t("action.delete")}</button>`,
      },
    ],
  });
  // Wrapper (Suche/Paging/Tabelle) initial ausblenden - erst loadDocuments()
  // (bei Auswahl einer Collection) blendet ihn wieder ein. Die "display:none"
  // am statischen <table>-Tag selbst greift nach der DataTables-Initialisierung
  // nicht mehr, da DataTables das <table> in einen eigenen Wrapper verschiebt.
  documentsTable.table().container().style.display = "none";
  // Löschen-Buttons werden bei jedem draw() (auch beim Seitenwechsel) neu
  // erzeugt - Listener daher hier statt einmalig binden. selectedCollection
  // (globaler State) statt eines geschlossenen Parameters, da draw() nicht
  // pro loadDocuments()-Aufruf neu gebunden wird.
  documentsTable.on("draw", () => {
    document.querySelectorAll("#documents-table .view-doc-btn").forEach(btn => {
      btn.addEventListener("click", () => openDocumentModal(selectedCollection, btn.dataset.id, btn.dataset.title));
    });
    document.querySelectorAll("#documents-table .del-doc-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        if (!confirm(t("confirm.deleteDocument"))) return;
        await fetch(`/rag/collections/${encodeURIComponent(selectedCollection)}/documents/${encodeURIComponent(btn.dataset.id)}`, {
          method: "DELETE", headers: authHeaders(),
        });
        loadDocuments(selectedCollection);
        loadCollections();
      });
    });
  });
}

async function loadDocuments(collection) {
  const res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/documents`, { headers: authHeaders() });
  const data = await res.json();
  const docs = data.documents || [];
  $("documents-empty-hint").style.display = "none";
  documentsTable.table().container().style.display = "";
  documentsTable.clear();
  documentsTable.rows.add(docs);
  documentsTable.draw(false);
}

// --- Add Content ---------------------------------------------------------
$("add-submit-btn").addEventListener("click", async () => {
  const collection = ($("add-collection-new").value || $("add-collection-select").value || "").trim();
  const statusEl = $("add-status");
  if (!collection) {
    statusEl.textContent = t("error.generic", { msg: "collection?" });
    statusEl.classList.add("error");
    return;
  }
  const isTextTab = $("tab-panel-text").classList.contains("active");
  const btn = $("add-submit-btn");
  btn.disabled = true;
  btn.textContent = t("action.adding");
  statusEl.classList.remove("error");
  statusEl.textContent = "";
  try {
    let res;
    if (isTextTab) {
      const text = $("add-text-content").value;
      const source = $("add-text-source").value || "text";
      res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/text`, {
        method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ text, source }),
      });
    } else {
      const path = $("add-file-path").value;
      res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/file`, {
        method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ path }),
      });
    }
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    statusEl.textContent = t("status.added", { n: data.chunks_added });
    $("add-text-content").value = "";
    $("add-file-path").value = "";
    $("add-collection-new").value = "";
    loadCollections();
    if (selectedCollection === collection) loadDocuments(collection);
  } catch (e) {
    statusEl.classList.add("error");
    statusEl.textContent = t("error.generic", { msg: e.message });
  } finally {
    btn.disabled = false;
    btn.textContent = t("action.add");
  }
});

// --- Test Search -----------------------------------------------------------
$("search-submit-btn").addEventListener("click", async () => {
  const collection = $("search-collection-select").value;
  const query = $("search-query").value;
  const topK = parseInt($("search-top-k").value, 10) || 5;
  const statusEl = $("search-status");
  if (!collection || !query) return;
  const btn = $("search-submit-btn");
  btn.disabled = true;
  btn.textContent = t("action.searching");
  statusEl.classList.remove("error");
  statusEl.textContent = "";
  try {
    const res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/search`, {
      method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderSearchResults(data.results || []);
  } catch (e) {
    statusEl.classList.add("error");
    statusEl.textContent = t("error.generic", { msg: e.message });
  } finally {
    btn.disabled = false;
    btn.textContent = t("action.search");
  }
});

function renderSearchResults(results) {
  if (results.length === 0) {
    $("search-results-box").innerHTML = `<div class="empty">${t("empty.noResults")}</div>`;
    return;
  }
  $("search-results-box").innerHTML = `<table><thead><tr>
    <th>${t("th.score")}</th><th>${t("th.text")}</th><th>${t("th.source")}</th>
    </tr></thead><tbody>` + results.map((r, i) => `
      <tr>
        <td class="mono">${(r.score ?? 0).toFixed(3)}<div class="score-bar-bg"><div class="score-bar-fg" style="width:${Math.max(0, Math.min(100, (r.score ?? 0) * 100))}%"></div></div></td>
        <td><div class="snippet" data-idx="${i}" title="${esc(t("action.viewFullText"))}">${esc((r.text || "").slice(0, 300))}${(r.text || "").length > 300 ? "…" : ""}</div></td>
        <td>${esc(r.source || "–")}</td>
      </tr>`).join("") + `</tbody></table>`;
  document.querySelectorAll("#search-results-box .snippet").forEach(el => {
    el.addEventListener("click", () => {
      const r = results[el.dataset.idx];
      openTextModal(r.source || "–", "", [{ chunk_index: r.chunk_index, chunk_count: r.chunk_count, text: r.text }]);
    });
  });
}

function refreshAll() {
  loadCollections();
  if (selectedCollection) loadDocuments(selectedCollection);
}

populateLangSelect();
applyStaticI18n();
initDocumentsTable();
loadCollections();
</script>
</body>
</html>
"""

RAG_DASHBOARD_HTML = RAG_DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
