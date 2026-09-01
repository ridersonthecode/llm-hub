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
<title>LLM Hub – RAG</title>
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.dataTables.min.css">
<link rel="stylesheet" href="/static/vendor/datatables/dataTables.inputPaging.min.css">
<style>
  :root {
    --bg:#f5f6f8; --panel:#ffffff; --panel-2:#eef0f4; --border:#dfe3ea;
    --text:#161922; --text-dim:#4b5363; --mono: "SF Mono", Consolas, "Liberation Mono", monospace;
    --accent:#2563eb; --good:#15803d; --warn:#b45309; --bad:#dc2626;
    --accent-bg:rgba(37,99,235,.10); --good-bg:rgba(21,128,61,.10); --bad-bg:rgba(220,38,38,.10); --warn-bg:rgba(180,83,9,.12);
    --code-bg:#0d1117; --code-text:#e6edf3; --code-bar:#161b22;
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
  /* Ganzes Dokument (alle Chunks der Reihe nach zu EINEM Text
     zusammengesetzt, siehe openTextModal/openDocumentModal) statt einer
     Chunk-für-Chunk-Ansicht - wer speichert, denkt in Dokumenten, nicht in
     den internen Speicher-Häppchen (Chat vom 2026-08-31). Umschaltbar
     zwischen Rohtext und gerendertem Markdown (für als Text eingefügte
     .md-Inhalte), siehe view-toggle unten. */
  .modal .view-toggle { display:flex; gap:6px; margin-bottom:14px; }
  .modal .view-toggle button {
    background:var(--panel); border:1px solid var(--border); color:var(--text-dim);
    border-radius:8px; height:30px; padding:0 12px; font-size:12px; cursor:pointer;
  }
  .modal .view-toggle button.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .modal .raw-text {
    background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
    padding:12px; font-family:var(--mono); font-size:12px; white-space:pre-wrap; word-break:break-word; margin:0;
  }
  .modal .md-render p { margin: 0 0 10px; }
  .modal .md-render p:last-child { margin-bottom:0; }
  .modal .md-render ul, .modal .md-render ol { margin: 4px 0 10px; padding-left: 22px; }
  .modal .md-render h3, .modal .md-render h4, .modal .md-render h5, .modal .md-render h6 { margin: 14px 0 8px; }
  .modal .md-render h3:first-child, .modal .md-render h4:first-child { margin-top:0; }
  .modal .md-render code { font-family: var(--mono); background:var(--panel-2); border-radius:4px; padding:1px 5px; font-size:.92em; }
  .modal .md-render .code-block { margin: 8px 0; border-radius:8px; overflow:hidden; border:1px solid var(--border); }
  .modal .md-render .code-block-bar {
    display:flex; justify-content:space-between; align-items:center; background:var(--code-bar); color:var(--text-dim);
    padding:4px 10px; font-size:11px;
  }
  .modal .md-render .code-block-bar button {
    background:none; border:none; color:var(--text-dim); cursor:pointer; font-size:11px; padding:2px 6px;
  }
  .modal .md-render .code-block-bar button:hover { color:#fff; }
  .modal .md-render .code-block pre { margin:0; background:var(--code-bg); color:var(--code-text); padding:12px; overflow-x:auto; }
  .modal .md-render .code-block code { font-family: var(--mono); font-size:12.5px; background:none; padding:0; }
  /* Ganzer Text steckt im DOM (fürs Sortieren/Suchen der DataTable), sichtbar
     aber auf eine Zeile gekürzt - voller Text per Browser-Tooltip (title-
     Attribut) oder vollständig/formatiert per "Anzeigen"-Button im Modal. */
  #documents-table .text-preview {
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; max-width:420px; white-space:normal; word-break:break-word;
    font-size:12.5px; color:var(--text-dim); line-height:1.4;
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
      <table id="documents-table" class="display" style="width:100%;"></table>
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
      <div class="view-toggle" id="text-modal-toggle">
        <button id="text-modal-view-md" data-i18n="modal.viewRendered">Rendered</button>
        <button id="text-modal-view-raw" data-i18n="modal.viewRaw">Raw text</button>
      </div>
      <pre class="raw-text" id="text-modal-raw" style="display:none;"></pre>
      <div class="md-render" id="text-modal-rendered" style="display:none;"></div>
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
  document.title = "LLM Hub – " + t("rag.title");
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

// --- Mini-Markdown-Renderer (identisch zum Chat-Dashboard, siehe dort) -----
// Bewusst kein CDN/keine externe Bibliothek. Nur für die Anzeige gedacht -
// wer als Text einen kompletten Markdown-Inhalt einfügt (Chat vom
// 2026-08-31: "ich möchte das auch als komplettes markdown wieder anschauen
// können"), soll ihn formatiert statt als Rohtext sehen können.
function mdRenderInline(text) {
  const codes = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(`<code>${esc(code)}</code>`);
    return `\u0000${codes.length - 1}\u0000`;
  });
  text = esc(text);
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  text = text.replace(/\u0000(\d+)\u0000/g, (_, i) => codes[Number(i)]);
  return text;
}
function mdRenderBlock(text) {
  const lines = text.split("\n");
  const htmlParts = [];
  let listBuf = [], listType = null, paraBuf = [];
  const flushList = () => {
    if (listBuf.length) {
      const tag = listType === "ol" ? "ol" : "ul";
      htmlParts.push(`<${tag}>${listBuf.map(li => `<li>${mdRenderInline(li)}</li>`).join("")}</${tag}>`);
      listBuf = []; listType = null;
    }
  };
  const flushPara = () => {
    if (paraBuf.length) { htmlParts.push(`<p>${mdRenderInline(paraBuf.join(" "))}</p>`); paraBuf = []; }
  };
  for (const line of lines) {
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    if (line.trim() === "") { flushPara(); flushList(); continue; }
    if (heading) {
      flushPara(); flushList();
      const level = heading[1].length + 2; // h3..h6 - h1/h2 bleiben der Modal-Überschrift vorbehalten
      htmlParts.push(`<h${level}>${mdRenderInline(heading[2])}</h${level}>`);
    } else if (ol) {
      flushPara(); if (listType !== "ol") flushList();
      listType = "ol"; listBuf.push(ol[1]);
    } else if (ul) {
      flushPara(); if (listType !== "ul") flushList();
      listType = "ul"; listBuf.push(ul[1]);
    } else {
      flushList(); paraBuf.push(line);
    }
  }
  flushPara(); flushList();
  return htmlParts.join("");
}
function mdRenderCodeBlock(lang, code) {
  const id = "rag-code-" + Math.random().toString(36).slice(2, 9);
  return `<div class="code-block">
    <div class="code-block-bar">
      <span>${esc(lang || "text")}</span>
      <button class="code-copy-btn" data-target="${id}">${esc(t("chat.action.copyCode"))}</button>
    </div>
    <pre><code id="${id}">${esc(code)}</code></pre>
  </div>`;
}
function renderMarkdown(text) {
  // An jedem ``` aufsplitten: gerade Indizes = normaler Text, ungerade = Code
  // (Sprache in der ersten Zeile) - siehe chat_dashboard.py für dieselbe Logik.
  const parts = (text || "").split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      html += mdRenderBlock(parts[i]);
    } else {
      const seg = parts[i];
      const nl = seg.indexOf("\n");
      const lang = nl === -1 ? seg.trim() : seg.slice(0, nl).trim();
      const code = nl === -1 ? "" : seg.slice(nl + 1);
      html += mdRenderCodeBlock(lang, code);
    }
  }
  return html;
}
// Grobe Heuristik, ob ein Text vermutlich Markdown ist - per Dateiendung
// (aus "Add File") oder anhand typischer Markdown-Syntax (aus "Add Text",
// wo es keine Dateiendung gibt). Bestimmt nur die Voreinstellung der
// Ansicht, per Klick jederzeit umschaltbar.
function looksLikeMarkdown(title, text) {
  if (/\.(md|markdown)$/i.test(title || "")) return true;
  if (!text) return false;
  return /^#{1,4}\s+\S/m.test(text) || /```/.test(text) || /^\s*[-*]\s+\S/m.test(text) || /\[[^\]]+\]\(https?:\/\//.test(text);
}
$("text-modal-rendered").addEventListener("click", async (e) => {
  const btn = e.target.closest(".code-copy-btn");
  if (!btn) return;
  const code = document.getElementById(btn.dataset.target)?.textContent || "";
  try {
    await navigator.clipboard.writeText(code);
    const orig = btn.textContent;
    btn.textContent = t("modal.copied");
    setTimeout(() => { btn.textContent = orig; }, 1200);
  } catch { /* Clipboard-API evtl. ohne HTTPS/Permission nicht verfügbar - stumm ignorieren. */ }
});

// --- Text-Detail-Modal ------------------------------------------------
// Zeigt ein Dokument oder einen Suchtreffer VOLLSTÄNDIG an - als EIN
// zusammenhängender Text (alle Chunks der Reihe nach zusammengesetzt), nicht
// Chunk für Chunk, da Chunking nur ein internes Speicher-Detail ist (Chat vom
// 2026-08-31). Umschaltbar zwischen Rohtext und gerendertem Markdown.
let modalFullText = "";
function setModalView(view) {
  $("text-modal-view-md").classList.toggle("active", view === "md");
  $("text-modal-view-raw").classList.toggle("active", view === "raw");
  $("text-modal-rendered").style.display = view === "md" ? "" : "none";
  $("text-modal-raw").style.display = view === "raw" ? "" : "none";
}
$("text-modal-view-md").addEventListener("click", () => setModalView("md"));
$("text-modal-view-raw").addEventListener("click", () => setModalView("raw"));

function openTextModal(title, hint, fullText, opts) {
  opts = opts || {};
  modalFullText = fullText || "";
  $("text-modal-title").textContent = title || "–";
  $("text-modal-hint").textContent = hint || "";
  $("text-modal-hint").style.display = hint ? "" : "none";
  $("text-modal-raw").textContent = modalFullText;
  $("text-modal-rendered").innerHTML = renderMarkdown(modalFullText);
  setModalView(opts.markdownGuess ? "md" : "raw");
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
  // Chunks kommen vom Backend schon nach chunk_index sortiert (siehe
  // rag.get_document_chunks) - hier nur noch zu einem Text zusammenfügen,
  // wie beim Speichern ursprünglich gechunkt wurde (chunk_text() bricht an
  // Absatz-/Satzgrenzen, siehe rag.py).
  const fullText = chunks.map(c => c.text || "").join("\n\n");
  const hint = chunks.length > 1 ? t("modal.chunkCountHint", { n: chunks.length }) : "";
  openTextModal(title, hint, fullText, { markdownGuess: looksLikeMarkdown(title, fullText) });
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
  await loadDocuments(selectedCollection);
}

function renderCollections() {
  if (collectionsCache.length === 0) {
    $("collections-box").innerHTML = `<div class="empty">${t("empty.noCollections")}</div>`;
    return;
  }
  // Erste Zeile "Alle Collections" (data-name="", selectedCollection===null)
  // fasst die Dokumente-Tabelle unten über alle Collections zusammen -
  // Standardansicht beim Laden, ganz ohne dass man erst eine Collection
  // anklicken oder die Testsuche bemühen muss (Chat vom 2026-08-31: "es gibt
  // keine generelle Seite wo ich alle Dokumente sehen kann").
  const totalPoints = collectionsCache.reduce((sum, c) => sum + (c.points_count || 0), 0);
  const allRow = `
      <tr class="clickable ${selectedCollection === null ? 'selected' : ''}" data-name="">
        <td><em>${esc(t("collections.all"))}</em></td>
        <td class="mono">${totalPoints}</td>
        <td></td>
      </tr>`;
  $("collections-box").innerHTML = `<table><thead><tr>
    <th>${t("th.collection")}</th><th>${t("th.chunks")}</th><th>${t("th.action")}</th>
    </tr></thead><tbody>` + allRow + collectionsCache.map(c => `
      <tr class="clickable ${c.name === selectedCollection ? 'selected' : ''}" data-name="${esc(c.name)}">
        <td>${esc(c.name)}</td>
        <td class="mono">${c.points_count}</td>
        <td><button class="btn danger del-collection-btn" data-name="${esc(c.name)}">${t("action.delete")}</button></td>
      </tr>`).join("") + `</tbody></table>`;

  $("collections-box").querySelectorAll("tr.clickable").forEach(tr => {
    tr.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      selectedCollection = tr.dataset.name || null;
      renderCollections();
      loadDocuments(selectedCollection);
    });
  });
  $("collections-box").querySelectorAll(".del-collection-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm(t("confirm.deleteCollection", { name: btn.dataset.name }))) return;
      await fetch(`/rag/collections/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE", headers: authHeaders() });
      if (selectedCollection === btn.dataset.name) selectedCollection = null;
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
      // Collection-Spalte: nur wirklich aussagekräftig in der "Alle
      // Collections"-Ansicht, aber auch bei einer gefilterten Collection
      // harmlos mitgeführt statt die Spalten je nach Ansicht neu aufzubauen.
      { title: t("th.collection"), data: null, render: (d, type) => esc(d.collection || "") },
      { title: t("th.source"), data: null, render: (d, type) => type === "display" ? esc(d.filename || d.source || "–") : (d.filename || d.source || "") },
      // Voller Text des Dokuments (siehe rag.list_documents - in Chunk-
      // Reihenfolge zusammengesetzt), gekürzt in der Zelle angezeigt
      // (CSS text-ellipsis-preview, per Klick/"Anzeigen"-Button vollständig
      // im Modal, siehe openDocumentModal) - Sortieren/Suchen laufen dank
      // orderData/type "display" trotzdem über den VOLLEN Text.
      {
        title: t("th.text"), data: null,
        render: (d, type) => type !== "display"
          ? (d.text || "")
          : `<span class="text-preview" title="${esc(d.text || "")}">${esc(d.text || "–")}</span>`,
      },
      { title: t("th.chunks"), data: null, render: (d, type) => type !== "display" ? (d.chunk_count ?? -1) : (d.chunk_count ?? "–") },
      { title: t("th.addedAt"), data: null, render: (d, type) => type !== "display" ? (d.added_at ?? 0) : (d.added_at ? new Date(d.added_at * 1000).toLocaleString(localeFor(currentLang)) : "–") },
      {
        title: t("th.action"), data: null, orderable: false,
        render: (d) => `<button class="view-doc-btn" data-collection="${esc(d.collection)}" data-id="${esc(d.document_id)}" data-title="${esc(d.filename || d.source || "–")}">${t("action.select")}</button>`
          + `<button class="btn danger del-doc-btn" data-collection="${esc(d.collection)}" data-id="${esc(d.document_id)}">${t("action.delete")}</button>`,
      },
    ],
  });
  // Löschen-Buttons werden bei jedem draw() (auch beim Seitenwechsel) neu
  // erzeugt - Listener daher hier statt einmalig binden. Collection kommt
  // aus data-collection (pro Zeile, siehe oben) statt aus dem globalen
  // selectedCollection, da die "Alle Collections"-Ansicht Zeilen aus
  // mehreren Collections gleichzeitig zeigt.
  documentsTable.on("draw", () => {
    document.querySelectorAll("#documents-table .view-doc-btn").forEach(btn => {
      btn.addEventListener("click", () => openDocumentModal(btn.dataset.collection, btn.dataset.id, btn.dataset.title));
    });
    // Gekürzte Textvorschau selbst anklickbar - derselbe Weg zum vollen
    // Text wie über den "Anzeigen"-Button, nur ohne extra Button-Treffer.
    document.querySelectorAll("#documents-table .text-preview").forEach(el => {
      el.style.cursor = "pointer";
      el.addEventListener("click", () => {
        const btn = el.closest("tr").querySelector(".view-doc-btn");
        if (btn) openDocumentModal(btn.dataset.collection, btn.dataset.id, btn.dataset.title);
      });
    });
    document.querySelectorAll("#documents-table .del-doc-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        if (!confirm(t("confirm.deleteDocument"))) return;
        await fetch(`/rag/collections/${encodeURIComponent(btn.dataset.collection)}/documents/${encodeURIComponent(btn.dataset.id)}`, {
          method: "DELETE", headers: authHeaders(),
        });
        loadCollections();
      });
    });
  });
}

async function fetchDocumentsFor(collection) {
  const res = await fetch(`/rag/collections/${encodeURIComponent(collection)}/documents`, { headers: authHeaders() });
  const data = await res.json();
  // Collection wird pro Dokument mitgeführt, weil die "Alle Collections"-
  // Ansicht (collection === null) Zeilen aus mehreren Collections mischt.
  return (data.documents || []).map(d => Object.assign({}, d, { collection }));
}

// collection === null -> über alle bekannten Collections hinweg (die neue
// "Alle Collections"-Zeile oben in der Collections-Tabelle) statt nur eine
// einzelne - Standardansicht, kein Klick auf eine Collection nötig.
async function loadDocuments(collection) {
  const docs = collection
    ? await fetchDocumentsFor(collection)
    : (await Promise.all(collectionsCache.map(c => fetchDocumentsFor(c.name))))
        .flat().sort((a, b) => (b.added_at || 0) - (a.added_at || 0));
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
  } catch (e) {
    statusEl.classList.add("error");
    statusEl.textContent = t("error.generic", { msg: e.message });
  } finally {
    btn.disabled = false;
    btn.textContent = t("action.add");
  }
});

// --- Test Search -----------------------------------------------------------
// Collection des zuletzt gestarteten Suchlaufs - renderSearchResults() (und
// darüber der Klick auf einen Treffer, der das ganze Dokument öffnet) kennt
// sonst nicht, aus welcher Collection die Ergebnisse stammen.
let lastSearchCollection = null;
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
    lastSearchCollection = collection;
    renderSearchResults(data.results || []);
  } catch (e) {
    statusEl.classList.add("error");
    statusEl.textContent = t("error.generic", { msg: e.message });
  } finally {
    btn.disabled = false;
    btn.textContent = t("action.search");
  }
});

// Die Suche findet einzelne Chunks, aber mehrere Treffer können aus
// demselben Dokument stammen - zu einer Zeile je Dokument zusammenfassen
// (bester Score gewinnt), da Chunking nur ein internes Speicher-Detail ist
// (Chat vom 2026-08-31: "ich möchte das immer zusammengefasst sehen und
// nicht die chunks einzeln", gleicher Gedanke wie bei der Dokumenten-Tabelle
// oben). Reihenfolge der Ergebnisse bleibt dabei erhalten.
function groupSearchResultsByDocument(results) {
  const grouped = [];
  const indexByKey = new Map();
  for (const r of results) {
    const key = r.document_id || `${r.source || ""}|${r.text || ""}`;
    if (indexByKey.has(key)) {
      const g = grouped[indexByKey.get(key)];
      g.matchCount++;
      if ((r.score ?? 0) > (g.score ?? 0)) { g.score = r.score; g.text = r.text; }
    } else {
      indexByKey.set(key, grouped.length);
      grouped.push(Object.assign({}, r, { matchCount: 1 }));
    }
  }
  grouped.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  return grouped;
}

function renderSearchResults(results) {
  if (results.length === 0) {
    $("search-results-box").innerHTML = `<div class="empty">${t("empty.noResults")}</div>`;
    return;
  }
  const grouped = groupSearchResultsByDocument(results);
  $("search-results-box").innerHTML = `<table><thead><tr>
    <th>${t("th.score")}</th><th>${t("th.text")}</th><th>${t("th.source")}</th>
    </tr></thead><tbody>` + grouped.map((r, i) => `
      <tr>
        <td class="mono">${(r.score ?? 0).toFixed(3)}<div class="score-bar-bg"><div class="score-bar-fg" style="width:${Math.max(0, Math.min(100, (r.score ?? 0) * 100))}%"></div></div></td>
        <td>
          <div class="snippet" data-idx="${i}" title="${esc(t("action.viewFullText"))}">${esc((r.text || "").slice(0, 300))}${(r.text || "").length > 300 ? "…" : ""}</div>
          ${r.matchCount > 1 ? `<div class="status-msg">${esc(t("search.matchCount", { n: r.matchCount }))}</div>` : ""}
        </td>
        <td>${esc(r.source || "–")}</td>
      </tr>`).join("") + `</tbody></table>`;
  document.querySelectorAll("#search-results-box .snippet").forEach(el => {
    el.addEventListener("click", () => {
      const r = grouped[el.dataset.idx];
      // Ganzes Dokument öffnen (alle Chunks zusammengesetzt) statt nur des
      // einen getroffenen Chunks - siehe openDocumentModal weiter oben.
      // Fallback auf den reinen Treffertext nur, falls document_id fehlt
      // (sollte bei aktuellen Daten nicht vorkommen).
      if (r.document_id && lastSearchCollection) {
        openDocumentModal(lastSearchCollection, r.document_id, r.source || "–");
      } else {
        openTextModal(r.source || "–", "", r.text || "", { markdownGuess: looksLikeMarkdown(r.source, r.text) });
      }
    });
  });
}

function refreshAll() {
  loadCollections();
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
