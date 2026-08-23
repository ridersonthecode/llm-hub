"""Config-Editor-Seite: /dashboard/config (HTML). Formularbasierte Bearbeitung
von config.json (Inputfelder/Dropdowns/Checkboxen statt rohem JSON), mit
Live-Übernahme oder komplettem Dienst-Neustart als Aktivierungsweg - siehe
config_editor.py für Validierung, automatisches Backup und den
Fallback-auf-letzte-funktionierende-Version-Mechanismus beim Programmstart.

Eigene, einfache Seite ohne WebSocket (kein Live-Push nötig, Config ändert
sich nur durch die eigene Aktion hier). Nutzt dieselben Übersetzungen wie das
Haupt-Dashboard (siehe dashboard.py)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .dashboard import _LANGUAGES_JS

router = APIRouter()


@router.get("/dashboard/config")
async def config_dashboard_page():
    return HTMLResponse(CONFIG_DASHBOARD_HTML)


CONFIG_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Manager – Config</title>
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
  .table-scroll { overflow-x:auto; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); vertical-align: top; }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  td.mono, th.mono { font-family: var(--mono); }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .banner { background:var(--warn-bg); color:var(--warn); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-size:13px; margin-bottom:16px; }
  .banner.info { background:var(--accent-bg); color:var(--accent); }
  label { display:block; font-size:12px; color:var(--text-dim); margin: 12px 0 4px; }
  label:first-child { margin-top: 0; }
  label .restart-badge { color:var(--warn); font-weight:600; margin-left:4px; }
  input[type=text], input[type=number], input[type=password], textarea, select {
    width:100%; background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:8px 10px; font-size:13px; font-family: inherit;
  }
  textarea { min-height: 70px; resize: vertical; font-family: var(--mono); }
  .row { display:flex; gap:12px; flex-wrap:wrap; }
  .row > div { flex: 1; min-width: 200px; }
  .check-row { display:flex; align-items:center; gap:8px; margin: 12px 0 4px; }
  .check-row input[type=checkbox] { width:16px; height:16px; }
  .check-row label { margin:0; font-size:13px; color:var(--text); }
  .hint { color:var(--text-dim); font-size:12px; margin-top:4px; }
  .action-bar {
    position:sticky; top:0; background:var(--bg); z-index:10; padding:12px 0;
    border-bottom:1px solid var(--border); margin-bottom:20px;
    display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  }
  .action-bar .spacer { flex:1; }
  .actions-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  .actions-row .spacer { flex:1; }
  .status-msg { font-size:12px; color:var(--text-dim); }
  .status-msg.error { color: var(--bad); white-space:pre-wrap; }
  .status-msg.ok { color: var(--good); }
  .dirty-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--warn); margin-left:6px; vertical-align:middle; }

  .accordion-item { background:var(--panel); border:1px solid var(--border); border-radius:10px; margin-bottom:10px; overflow:hidden; }
  .accordion-header {
    display:flex; align-items:center; gap:10px; padding:12px 14px; cursor:pointer; user-select:none;
  }
  .accordion-header:hover { background:var(--panel-2); }
  .accordion-header .name { font-family: var(--mono); font-size:13px; font-weight:600; flex:1; word-break: break-all; }
  .accordion-header .chevron { color:var(--text-dim); transition: transform .15s; font-size:12px; }
  .accordion-item.open .chevron { transform: rotate(90deg); }
  .accordion-body { display:none; padding: 4px 14px 16px; border-top:1px solid var(--border); }
  .accordion-item.open .accordion-body { display:block; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.ok { background: var(--good-bg); color: var(--good); }
  .badge.idle { background: var(--panel-2); color: var(--text-dim); }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1 data-i18n="cfg.title">Configuration</h1>
      <div class="sub"><a href="/dashboard" data-i18n="nav.dashboardLink">← Dashboard</a></div>
    </div>
    <div class="topbar-actions">
      <select id="lang-select" data-i18n-title="lang.selectTitle" title="Language"></select>
      <button id="theme-toggle" data-i18n-title="theme.toggleTitle" title="Toggle theme">🌙</button>
    </div>
  </div>

  <div id="startup-warning-banner" class="banner" style="display:none;"></div>
  <div class="banner info" data-i18n="cfg.hint.liveVsRestart">
    Most settings apply immediately after saving. Already-loaded models keep their previous settings until reloaded. Only host/port require a full service restart.
  </div>

  <div class="action-bar">
    <button class="btn primary" id="save-btn" data-i18n="cfg.action.save">Save (apply live)</button>
    <button class="btn" id="save-restart-btn" data-i18n="cfg.action.saveRestart">Save &amp; restart service</button>
    <button class="btn" id="discard-btn" data-i18n="cfg.action.discard">Discard changes</button>
    <span id="dirty-indicator" style="display:none;"><span class="dirty-dot"></span><span class="status-msg" data-i18n="cfg.status.unsavedChanges">Unsaved changes</span></span>
    <div class="spacer"></div>
    <span class="status-msg" id="save-status"></span>
  </div>

  <section>
    <h2 data-i18n="cfg.section.server">Server</h2>
    <div class="card">
      <div class="row">
        <div>
          <label data-i18n="cfg.field.host">Host<span class="restart-badge" data-i18n="cfg.restartBadge">🔁 restart</span></label>
          <input type="text" id="f-host">
        </div>
        <div>
          <label data-i18n="cfg.field.port">Port<span class="restart-badge" data-i18n="cfg.restartBadge">🔁 restart</span></label>
          <input type="number" id="f-port" min="1" max="65535">
        </div>
      </div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.engineHost">Engine Host</label>
          <input type="text" id="f-engine_host">
        </div>
        <div>
          <label data-i18n="cfg.field.enginePort">Engine Base Port</label>
          <input type="number" id="f-engine_port" min="1" max="65535">
        </div>
      </div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.hfHome">HF Home (model cache dir)</label>
          <input type="text" id="f-hf_home">
        </div>
        <div>
          <label data-i18n="cfg.field.vllmBin">vLLM Binary (empty = auto)</label>
          <input type="text" id="f-vllm_bin" placeholder="auto">
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.apiKey">API Key</h2>
    <div class="card">
      <div class="check-row">
        <input type="checkbox" id="f-api_key_enabled">
        <label for="f-api_key_enabled" data-i18n="cfg.field.apiKeyEnabled">Require API key</label>
      </div>
      <label data-i18n="cfg.field.apiKeyKey">Key</label>
      <input type="text" id="f-api_key_key" autocomplete="off">
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.hotPool">Hot Pool &amp; Limits</h2>
    <div class="card">
      <div class="row">
        <div>
          <label data-i18n="cfg.field.maxConcurrentModels">Max concurrent models (hot pool size)</label>
          <input type="number" id="f-max_concurrent_models" min="1" max="16">
        </div>
        <div>
          <label data-i18n="cfg.field.gpuMemoryCeiling">GPU memory ceiling (sum of all engines)</label>
          <input type="number" id="f-gpu_memory_ceiling" min="0" max="1" step="0.01">
        </div>
      </div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.idleTimeout">Idle timeout (seconds)</label>
          <input type="number" id="f-idle_timeout_seconds" min="0">
          <div class="hint" data-i18n="cfg.hint.idleTimeout">Empty = never unload automatically</div>
        </div>
        <div>
          <label data-i18n="cfg.field.startupTimeout">Startup timeout (seconds)</label>
          <input type="number" id="f-startup_timeout_seconds" min="1">
        </div>
      </div>
      <label data-i18n="cfg.field.defaultModel">Default model (used when a request has no "model")</label>
      <select id="f-default_model"></select>

      <div class="check-row">
        <input type="checkbox" id="f-auto_reload_last_model">
        <label for="f-auto_reload_last_model" data-i18n="cfg.field.autoReloadLastModel">Auto-reload last used model on service restart</label>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.defaultServeArgs">Default Serve Args</h2>
    <div class="card">
      <div class="hint" data-i18n="cfg.hint.defaultServeArgs">Fallback values for models without their own override below.</div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.defaultGpuMemUtil">GPU memory utilization</label>
          <input type="number" id="f-dsa_gpu_memory_utilization" min="0" max="1" step="0.01">
        </div>
        <div>
          <label data-i18n="cfg.field.defaultMaxModelLen">Max model length</label>
          <input type="number" id="f-dsa_max_model_len" min="1">
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.pricing">Cost Tracking</h2>
    <div class="card">
      <div class="hint" data-i18n="cfg.hint.pricing">Fictional prices for cost comparison (see the Costs page) - default is standard Claude Sonnet 5 pricing. Purely informational, no effect on actual (free) local operation. Per-model overrides are set on each model below.</div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.pricingInput">Input $ / MTok</label>
          <input type="number" id="f-pricing_input_per_mtok" min="0" step="0.01">
        </div>
        <div>
          <label data-i18n="cfg.field.pricingOutput">Output $ / MTok</label>
          <input type="number" id="f-pricing_output_per_mtok" min="0" step="0.01">
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.rag">RAG</h2>
    <div class="card">
      <div class="check-row">
        <input type="checkbox" id="f-rag_enabled">
        <label for="f-rag_enabled" data-i18n="cfg.field.ragEnabled">Enable RAG</label>
      </div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.qdrantHost">Qdrant Host</label>
          <input type="text" id="f-rag_qdrant_host">
        </div>
        <div>
          <label data-i18n="cfg.field.qdrantPort">Qdrant Port</label>
          <input type="number" id="f-rag_qdrant_port" min="1" max="65535">
        </div>
      </div>
      <label data-i18n="cfg.field.embeddingModel">Embedding model</label>
      <select id="f-rag_embedding_model"></select>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.defaultCollection">Default collection</label>
          <input type="text" id="f-rag_default_collection">
        </div>
      </div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.chunkSize">Chunk size (chars)</label>
          <input type="number" id="f-rag_chunk_size_chars" min="1">
        </div>
        <div>
          <label data-i18n="cfg.field.chunkOverlap">Chunk overlap (chars)</label>
          <input type="number" id="f-rag_chunk_overlap_chars" min="0">
        </div>
      </div>
      <div class="hint" data-i18n="cfg.hint.autoRag">Settings below apply to automatic server-side RAG (see "Auto-RAG collection" on each model further down) - not to manual searches via the RAG page/API, which take their own top_k per request.</div>
      <div class="row">
        <div>
          <label data-i18n="cfg.field.autoRagTopK">Auto-RAG: matches per request</label>
          <input type="number" id="f-rag_auto_rag_top_k" min="1" max="20">
        </div>
        <div>
          <label data-i18n="cfg.field.autoRagMinScore">Auto-RAG: minimum relevance score</label>
          <input type="number" id="f-rag_auto_rag_min_score" min="0" max="1" step="0.05">
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.models">Models</h2>
    <div id="models-box"></div>
    <div style="margin-top:10px;">
      <button class="btn" id="add-model-btn" data-i18n="cfg.action.addModel">+ Add model</button>
    </div>
  </section>

  <section>
    <h2 data-i18n="cfg.section.backups">Backups</h2>
    <div id="backups-box"></div>
  </section>

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
  document.title = "vLLM Manager – " + t("cfg.title");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
}
$("lang-select").addEventListener("change", (e) => {
  currentLang = e.target.value;
  localStorage.setItem("vllm_dashboard_lang", currentLang);
  applyStaticI18n();
  renderModels();
  renderBackups();
});

// --- Theme -------------------------------------------------------------
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
async function ensureApiKey() {
  if (apiKey) return;
  const key = prompt(t("auth.apiKeyPrompt"));
  if (key) { apiKey = key; sessionStorage.setItem("vllm_dashboard_key", key); }
}

// --- State ---------------------------------------------------------------
// `state` ist die Arbeitskopie der Config, direkt an die Formularfelder
// gebunden. `modelsList` ist dieselben Modelle als Array (Reihenfolge +
// Edit-UI einfacher als über das dict direkt) - wird bei jeder Änderung nach
// state.models zurückgeschrieben.
let state = null;
let modelsList = [];
let dirty = false;
let openAccordions = new Set();

function markDirty() {
  dirty = true;
  $("dirty-indicator").style.display = "inline";
}
function clearDirty() {
  dirty = false;
  $("dirty-indicator").style.display = "none";
}

let ragCollectionNames = [];

async function loadConfig() {
  const res = await fetch("/config", { headers: authHeaders() });
  if (res.status === 401) { await ensureApiKey(); return loadConfig(); }
  const data = await res.json();
  state = data.config;
  modelsList = Object.entries(state.models || {}).map(([name, m]) => Object.assign({ name }, m));
  // Alphabetisch (case-insensitive) - Backend liefert cfg.models bereits sortiert
  // (siehe config.py: sort_models()), hier zusätzlich sortiert für den Fall, dass
  // sich das mal ändert bzw. für Robustheit unabhängig vom Backend.
  modelsList.sort((a, b) => (a.name || "").localeCompare(b.name || "", undefined, { sensitivity: "base" }));
  if (data.startup_warning) {
    $("startup-warning-banner").style.display = "block";
    $("startup-warning-banner").textContent = "⚠️ " + data.startup_warning;
  } else {
    $("startup-warning-banner").style.display = "none";
  }
  // Für das Autovervollständigungs-Datalist beim "Auto-RAG collection"-Feld -
  // rein informativ (Freitextfeld bleibt trotzdem editierbar), schlägt still
  // fehl, falls RAG gar nicht konfiguriert ist (400 von /rag/collections).
  try {
    const ragRes = await fetch("/rag/collections", { headers: authHeaders() });
    if (ragRes.ok) {
      const ragData = await ragRes.json();
      ragCollectionNames = (ragData.collections || []).map(c => c.name);
    }
  } catch (e) { /* RAG nicht konfiguriert - Datalist bleibt einfach leer */ }
  renderForm();
  renderModels();
  clearDirty();
  await loadBackups();
}

function bindText(id, path) {
  const el = $(id);
  el.value = path() ?? "";
  el.addEventListener("input", () => { markDirty(); });
}

function renderForm() {
  $("f-host").value = state.host ?? "";
  $("f-port").value = state.port ?? "";
  $("f-engine_host").value = state.engine_host ?? "";
  $("f-engine_port").value = state.engine_port ?? "";
  $("f-hf_home").value = state.hf_home ?? "";
  $("f-vllm_bin").value = state.vllm_bin ?? "";

  $("f-api_key_enabled").checked = !!(state.api_key && state.api_key.enabled);
  $("f-api_key_key").value = (state.api_key && state.api_key.key) || "";

  $("f-max_concurrent_models").value = state.max_concurrent_models ?? 1;
  $("f-gpu_memory_ceiling").value = state.gpu_memory_ceiling ?? 0.9;
  $("f-idle_timeout_seconds").value = state.idle_timeout_seconds ?? "";
  $("f-startup_timeout_seconds").value = state.startup_timeout_seconds ?? 900;

  const dsa = state.default_serve_args || {};
  $("f-dsa_gpu_memory_utilization").value = dsa.gpu_memory_utilization ?? "";
  $("f-dsa_max_model_len").value = dsa.max_model_len ?? "";

  const pricing = state.default_pricing || {};
  $("f-pricing_input_per_mtok").value = pricing.input_per_mtok ?? 3.0;
  $("f-pricing_output_per_mtok").value = pricing.output_per_mtok ?? 15.0;

  const rag = state.rag || {};
  $("f-rag_enabled").checked = !!rag.enabled;
  $("f-rag_qdrant_host").value = rag.qdrant_host ?? "127.0.0.1";
  $("f-rag_qdrant_port").value = rag.qdrant_port ?? 6333;
  $("f-rag_default_collection").value = rag.default_collection ?? "default";
  $("f-rag_chunk_size_chars").value = rag.chunk_size_chars ?? 1500;
  $("f-rag_chunk_overlap_chars").value = rag.chunk_overlap_chars ?? 200;
  $("f-rag_auto_rag_top_k").value = rag.auto_rag_top_k ?? 3;
  $("f-rag_auto_rag_min_score").value = rag.auto_rag_min_score ?? 0.5;

  populateModelSelects();
  $("f-default_model").value = state.default_model ?? "";
  $("f-auto_reload_last_model").checked = state.auto_reload_last_model !== false;
  $("f-rag_embedding_model").value = rag.embedding_model ?? "";

  document.querySelectorAll("#f-host,#f-port,#f-engine_host,#f-engine_port,#f-hf_home,#f-vllm_bin,"
    + "#f-api_key_enabled,#f-api_key_key,#f-max_concurrent_models,#f-gpu_memory_ceiling,"
    + "#f-idle_timeout_seconds,#f-startup_timeout_seconds,#f-auto_reload_last_model,"
    + "#f-dsa_gpu_memory_utilization,#f-dsa_max_model_len,"
    + "#f-pricing_input_per_mtok,#f-pricing_output_per_mtok,"
    + "#f-rag_enabled,#f-rag_qdrant_host,#f-rag_qdrant_port,#f-rag_default_collection,"
    + "#f-rag_chunk_size_chars,#f-rag_chunk_overlap_chars,#f-rag_auto_rag_top_k,#f-rag_auto_rag_min_score,"
    + "#f-default_model,#f-rag_embedding_model")
    .forEach(el => { el.oninput = markDirty; el.onchange = markDirty; });
}

function populateModelSelects() {
  // Wird bei JEDER Namens-/Task-Änderung oder jedem Entfernen eines
  // BELIEBIGEN Modells neu aufgerufen (siehe Aufrufer unten), nicht nur wenn
  // das ausgewählte Modell selbst betroffen ist. Ohne die aktuelle Auswahl
  // hier zu merken und wiederherzustellen, wurde <select> komplett neu
  // aufgebaut und damit die Auswahl STILLSCHWEIGEND auf "kein Modell"
  // zurückgesetzt - z.B. reichte das Umbenennen eines völlig anderen Modells,
  // um default_model bzw. rag.embedding_model beim nächsten Speichern
  // versehentlich auf null zu setzen (RAG damit faktisch deaktiviert).
  const prevDefault = $("f-default_model").value;
  const prevEmbed = $("f-rag_embedding_model").value;
  const names = modelsList.map(m => m.name).filter(Boolean);
  const embedNames = modelsList.filter(m => m.task === "embed").map(m => m.name);
  const opt = (v) => `<option value="${esc(v)}">${esc(v)}</option>`;
  $("f-default_model").innerHTML = `<option value="">${esc(t("cfg.hint.noModel"))}</option>` + names.map(opt).join("");
  $("f-rag_embedding_model").innerHTML = `<option value="">${esc(t("cfg.hint.noModel"))}</option>` + embedNames.map(opt).join("");
  if (names.includes(prevDefault)) $("f-default_model").value = prevDefault;
  if (embedNames.includes(prevEmbed)) $("f-rag_embedding_model").value = prevEmbed;
}

// --- Models: Accordion-Liste ----------------------------------------------
function renderModels() {
  const box = $("models-box");
  if (modelsList.length === 0) {
    box.innerHTML = `<div class="empty">${t("empty.noModelsKnown")}</div>`;
    return;
  }
  box.innerHTML = modelsList.map((m, i) => {
    const open = openAccordions.has(i);
    return `
    <div class="accordion-item ${open ? "open" : ""}" data-idx="${i}">
      <div class="accordion-header" data-toggle="${i}">
        <span class="chevron">▶</span>
        <span class="name">${esc(m.name || t("cfg.hint.unnamedModel"))}</span>
        <span class="badge ${m.enabled === false ? "idle" : "ok"}">${m.enabled === false ? t("badge.disabled") : t("badge.enabled")}</span>
        ${m.vision ? `<span class="badge idle">${t("badge.vision")}</span>` : ""}
        ${m.task === "embed" ? `<span class="badge idle">embed</span>` : ""}
      </div>
      <div class="accordion-body">
        <label data-i18n="cfg.field.modelName">Model name / path</label>
        <input type="text" class="m-field" data-idx="${i}" data-field="name" value="${esc(m.name)}">

        <div class="check-row">
          <input type="checkbox" class="m-field" data-idx="${i}" data-field="enabled" id="m-enabled-${i}" ${m.enabled !== false ? "checked" : ""}>
          <label for="m-enabled-${i}" data-i18n="cfg.field.modelEnabled">Enabled</label>
        </div>

        <div style="margin: 12px 0;">
          <button class="btn detect-btn" data-idx="${i}" data-i18n="cfg.action.detectCapabilities">🔍 Auto-detect capabilities</button>
          <div class="detect-results" id="detect-results-${i}" style="display:none;"></div>
        </div>

        <label data-i18n="cfg.field.task">Task</label>
        <select class="m-field" data-idx="${i}" data-field="task">
          <option value="generate" ${m.task !== "embed" ? "selected" : ""}>generate</option>
          <option value="embed" ${m.task === "embed" ? "selected" : ""}>embed</option>
        </select>

        <div class="row">
          <div>
            <label data-i18n="cfg.field.maxModelLen">Max model length</label>
            <input type="number" class="m-field" data-idx="${i}" data-field="max_model_len" min="1" value="${m.max_model_len ?? ""}">
          </div>
          <div>
            <label data-i18n="cfg.field.gpuMemUtil">GPU memory utilization</label>
            <input type="number" class="m-field" data-idx="${i}" data-field="gpu_memory_utilization" min="0" max="1" step="0.01" value="${m.gpu_memory_utilization ?? ""}">
          </div>
        </div>

        <label data-i18n="cfg.field.maxTokens">Max output tokens (safety net, empty = unbounded)</label>
        <input type="number" class="m-field" data-idx="${i}" data-field="max_tokens" min="1" value="${m.max_tokens ?? ""}">
        <div class="hint" data-i18n="cfg.hint.maxTokens">Caps generation length only when the client itself doesn't request a max_tokens value - protects against runaway/repeating generations without limiting normal replies.</div>

        <label data-i18n="cfg.field.ragCollection">Auto-RAG collection (empty = off)</label>
        <input type="text" class="m-field" data-idx="${i}" data-field="rag_collection" list="rag-collection-options" value="${esc(m.rag_collection ?? "")}">
        <div class="hint" data-i18n="cfg.hint.ragCollection">If set, every chat request to this model automatically searches this collection and prepends relevant context - the client doesn't need to support anything special. Requires RAG to be enabled with an embedding model configured (see the RAG section below).</div>

        <label data-i18n="cfg.field.repetitionPenalty">Repetition penalty (empty = off)</label>
        <input type="number" class="m-field" data-idx="${i}" data-field="repetition_penalty" min="1" step="0.05" value="${m.repetition_penalty ?? ""}">
        <div class="hint" data-i18n="cfg.hint.repetitionPenalty">Only applied when the client itself doesn't set repetition_penalty. Reduces the chance of runaway repetition loops (common with smaller reasoning models). Try 1.1-1.3 if a model tends to get stuck.</div>

        <div class="check-row">
          <input type="checkbox" class="m-field" data-idx="${i}" data-field="repetition_detection" id="m-repdet-${i}" ${m.repetition_detection !== false ? "checked" : ""}>
          <label for="m-repdet-${i}" data-i18n="cfg.field.repetitionDetection">Detect and abort repetition loops (streamed requests)</label>
        </div>
        <div class="hint" data-i18n="cfg.hint.repetitionDetection">Watches the live stream and aborts the request if the same text keeps repeating verbatim - shows up in Active/Recent Requests as "Aborted (loop)". Disable only if a model is expected to legitimately repeat text.</div>

        <div class="row">
          <div>
            <label data-i18n="cfg.field.toolCallParser">Tool call parser</label>
            <input type="text" class="m-field" data-idx="${i}" data-field="tool_call_parser" list="tool-parser-options" value="${esc(m.tool_call_parser ?? "")}">
          </div>
          <div>
            <label data-i18n="cfg.field.reasoningParser">Reasoning parser</label>
            <input type="text" class="m-field" data-idx="${i}" data-field="reasoning_parser" list="reasoning-parser-options" value="${esc(m.reasoning_parser ?? "")}">
          </div>
        </div>

        <div class="check-row">
          <input type="checkbox" class="m-field" data-idx="${i}" data-field="enable_auto_tool_choice" id="m-tool-${i}" ${m.enable_auto_tool_choice ? "checked" : ""}>
          <label for="m-tool-${i}" data-i18n="cfg.field.autoToolChoice">Enable auto tool choice</label>
        </div>
        <div class="check-row">
          <input type="checkbox" class="m-field" data-idx="${i}" data-field="vision" id="m-vision-${i}" ${m.vision ? "checked" : ""}>
          <label for="m-vision-${i}" data-i18n="badge.vision">Vision</label>
        </div>

        <div class="row">
          <div>
            <label data-i18n="cfg.field.pricingInputOverride">Cost tracking: input $/MTok (empty = default)</label>
            <input type="number" class="m-field" data-idx="${i}" data-field="pricing_input_per_mtok" min="0" step="0.01" value="${(m.pricing && m.pricing.input_per_mtok != null) ? m.pricing.input_per_mtok : ""}">
          </div>
          <div>
            <label data-i18n="cfg.field.pricingOutputOverride">Cost tracking: output $/MTok (empty = default)</label>
            <input type="number" class="m-field" data-idx="${i}" data-field="pricing_output_per_mtok" min="0" step="0.01" value="${(m.pricing && m.pricing.output_per_mtok != null) ? m.pricing.output_per_mtok : ""}">
          </div>
        </div>

        <label data-i18n="cfg.field.hfToken">HF token (optional, for gated models)</label>
        <input type="text" class="m-field" data-idx="${i}" data-field="hf_token" value="${esc(m.hf_token ?? "")}">

        <label data-i18n="cfg.field.extraArgs">Extra args (one per line)</label>
        <textarea class="m-field" data-idx="${i}" data-field="extra_args">${esc((m.extra_args || []).join("\n"))}</textarea>
        <div class="hint" data-i18n="cfg.hint.extraArgs">One flag/value per line, e.g. --foo then bar on the next line.</div>

        <label data-i18n="cfg.field.notes">Notes</label>
        <textarea class="m-field" data-idx="${i}" data-field="notes">${esc(m.notes ?? "")}</textarea>

        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
          <button class="btn danger m-remove" data-idx="${i}" data-i18n="action.delete">Remove</button>
          <button class="btn danger m-delete-cache" data-name="${esc(m.name || "")}" data-i18n="action.deleteFromDisk">Delete from disk</button>
        </div>
      </div>
    </div>`;
  }).join("") + `
    <datalist id="tool-parser-options">
      <option value="hermes"><option value="qwen3_xml"><option value="qwen3_coder">
      <option value="openai"><option value="llama3_json"><option value="mistral">
      <option value="granite">
    </datalist>
    <datalist id="reasoning-parser-options">
      <option value="qwen3"><option value="deepseek_r1"><option value="deepseek_v3">
      <option value="granite"><option value="mistral"><option value="openai_gptoss">
      <option value="hunyuan_a13b"><option value="glm45"><option value="glm47">
    </datalist>
    <datalist id="rag-collection-options">${ragCollectionNames.map(n => `<option value="${esc(n)}">`).join("")}</datalist>`;

  document.querySelectorAll(".accordion-header").forEach(el => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.dataset.toggle, 10);
      if (openAccordions.has(idx)) openAccordions.delete(idx); else openAccordions.add(idx);
      el.closest(".accordion-item").classList.toggle("open");
    });
  });
  document.querySelectorAll(".m-field").forEach(el => {
    const handler = () => {
      const idx = parseInt(el.dataset.idx, 10);
      const field = el.dataset.field;
      let val;
      if (el.type === "checkbox") val = el.checked;
      else if (field === "extra_args") val = el.value.split("\n").map(s => s.trim()).filter(Boolean);
      else if (field === "max_model_len" || field === "max_tokens") val = el.value === "" ? null : parseInt(el.value, 10);
      else if (field === "gpu_memory_utilization" || field === "repetition_penalty") val = el.value === "" ? null : parseFloat(el.value);
      else val = el.value;
      if (field === "pricing_input_per_mtok" || field === "pricing_output_per_mtok") {
        // Verschachteltes Feld (ModelConfig.pricing.*) statt eines flachen -
        // beide leer = kompletter Override entfernt (null -> default_pricing gilt).
        const sub = field === "pricing_input_per_mtok" ? "input_per_mtok" : "output_per_mtok";
        const num = el.value === "" ? null : parseFloat(el.value);
        const pricing = Object.assign({}, modelsList[idx].pricing || {});
        if (num === null) delete pricing[sub]; else pricing[sub] = num;
        modelsList[idx].pricing = Object.keys(pricing).length ? pricing : null;
      } else {
        modelsList[idx][field] = val === "" && (field === "tool_call_parser" || field === "reasoning_parser" || field === "hf_token" || field === "notes" || field === "rag_collection") ? null : val;
      }
      markDirty();
      if (field === "name" || field === "task") { populateModelSelects(); }
      if (field === "name") {
        const header = document.querySelector(`.accordion-item[data-idx="${idx}"] .name`);
        if (header) header.textContent = val || t("cfg.hint.unnamedModel");
      }
    };
    el.addEventListener("input", handler);
    el.addEventListener("change", handler);
  });
  document.querySelectorAll(".m-remove").forEach(el => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.dataset.idx, 10);
      const name = modelsList[idx].name || t("cfg.hint.unnamedModel");
      if (!confirm(t("confirm.removeModel", { model: name }))) return;
      modelsList.splice(idx, 1);
      openAccordions = new Set();
      markDirty();
      renderModels();
      populateModelSelects();
    });
  });
  document.querySelectorAll(".m-delete-cache").forEach(el => {
    el.addEventListener("click", () => deleteModelCacheFromEditor(el.dataset.name, el));
  });
  document.querySelectorAll(".detect-btn").forEach(el => {
    el.addEventListener("click", () => detectCapabilities(parseInt(el.dataset.idx, 10)));
  });
}

// --- Lokale Modell-Dateien unwiderruflich von der Platte löschen ---------
// Bewusst UNABHÄNGIG von "Remove"/Save/Discard hier oben: löscht sofort und
// endgültig, egal ob der Modell-Eintrag anschließend gespeichert oder
// verworfen wird - siehe main.py DELETE /models/{model}/cache. Arbeitet über
// den Modellnamen, nicht den Array-Index, funktioniert also auch für ein
// bereits per "Remove" aus modelsList entferntes (aber lokal noch gecachtes)
// Modell, solange man dessen Namen kennt.
async function deleteModelCacheFromEditor(name, btn) {
  name = (name || "").trim();
  if (!name) {
    alert(t("cfg.status.detectNoName"));
    return;
  }
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t("action.checkingSize");
  try {
    const infoRes = await fetch(`/models/${encodeURIComponent(name)}/cache_info`, { headers: authHeaders() });
    if (!infoRes.ok) throw new Error(await infoRes.text());
    const info = await infoRes.json();
    if (!info.cached) {
      alert(t("info.nothingCachedToDelete", { model: name }));
      btn.disabled = false;
      btn.textContent = original;
      return;
    }
    const gb = (info.size_bytes / 1e9).toFixed(1);
    if (!confirm(t("confirm.deleteModelCache", { model: name, size: gb }))) {
      btn.disabled = false;
      btn.textContent = original;
      return;
    }
    btn.textContent = t("action.deleting");
    const delRes = await fetch(`/models/${encodeURIComponent(name)}/cache`, { method: "DELETE", headers: authHeaders() });
    if (!delRes.ok) throw new Error(await delRes.text());
    btn.textContent = original;
    btn.disabled = false;
  } catch (e) {
    alert(t("error.deleteCacheFailed", { msg: e.message }));
    btn.disabled = false;
    btn.textContent = original;
  }
}

// --- Fähigkeiten automatisch erkennen (chat_template.jinja/config.json aus
// dem lokalen HF-Cache, siehe capability_detector.py) - reiner Vorschlag,
// wird erst nach Klick auf "Übernehmen" in die Formularfelder geschrieben. ---
async function detectCapabilities(idx) {
  const name = (modelsList[idx].name || "").trim();
  const box = $(`detect-results-${idx}`);
  if (!box) return;
  if (!name) {
    box.style.display = "block";
    box.innerHTML = `<div class="hint">${esc(t("cfg.status.detectNoName"))}</div>`;
    return;
  }
  box.style.display = "block";
  box.innerHTML = `<div class="hint">${esc(t("cfg.status.detecting"))}</div>`;
  try {
    const res = await fetch(`/models/${encodeURIComponent(name)}/detect_capabilities`, { headers: authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    if (!data.found) {
      box.innerHTML = `<div class="hint">${esc(t("cfg.status.detectNotCached"))}</div>`;
      return;
    }
    renderDetectResults(idx, data);
  } catch (e) {
    box.innerHTML = `<div class="hint" style="color:var(--bad)">${esc(t("cfg.status.detectFailed", { msg: e.message }))}</div>`;
  }
}

function renderDetectResults(idx, data) {
  const box = $(`detect-results-${idx}`);
  const confBadge = (c) => {
    if (c === "high") return `<span class="badge ok">${t("cfg.detect.confHigh")}</span>`;
    if (c === "low") return `<span class="badge idle">${t("cfg.detect.confLow")}</span>`;
    return "";
  };
  const valueCell = (v) => v === true ? "✅" : v === false ? "–" : esc(v ?? "–");
  const rows = [
    [t("badge.vision"), valueCell(data.vision.detected), data.vision.confidence, data.vision.evidence],
    [t("badge.toolCalling"), valueCell(data.tool_calling.detected ? (data.tool_calling.suggested_parser || true) : false), data.tool_calling.confidence, data.tool_calling.evidence],
    [t("badge.reasoning"), valueCell(data.reasoning.detected ? (data.reasoning.suggested_parser || true) : false), data.reasoning.confidence, data.reasoning.evidence],
    [t("cfg.field.task"), valueCell(data.task.suggested), data.task.confidence, data.task.evidence],
  ];
  box.innerHTML = `
    <div class="card" style="margin-top:8px;">
      <table><tbody>
        ${rows.map(([label, val, conf, evidence]) => `<tr>
          <td>${esc(label)}</td>
          <td class="mono">${val}</td>
          <td>${confBadge(conf)}</td>
          <td class="hint">${esc(evidence || "")}</td>
        </tr>`).join("")}
      </tbody></table>
      <div class="actions-row" style="margin-top:10px;">
        <span class="hint">${t("cfg.hint.detectDisclaimer")}</span>
        <div class="spacer"></div>
        <button class="btn primary apply-detect-btn" data-idx="${idx}">${t("cfg.action.applyDetected")}</button>
      </div>
    </div>`;
  box.querySelector(".apply-detect-btn").addEventListener("click", () => applyDetected(idx, data));
}

function applyDetected(idx, data) {
  const m = modelsList[idx];
  m.vision = !!data.vision.detected;
  m.enable_auto_tool_choice = !!data.tool_calling.detected;
  m.tool_call_parser = data.tool_calling.detected ? (data.tool_calling.suggested_parser || m.tool_call_parser) : null;
  m.reasoning_parser = data.reasoning.detected ? (data.reasoning.suggested_parser || m.reasoning_parser) : null;
  m.task = data.task.suggested;
  markDirty();
  openAccordions.add(idx);
  renderModels();
}

$("add-model-btn").addEventListener("click", () => {
  modelsList.push({
    name: "", enabled: true, task: "generate", max_model_len: null,
    gpu_memory_utilization: null, tool_call_parser: null, reasoning_parser: null,
    enable_auto_tool_choice: false, vision: false, extra_args: [], hf_token: null, notes: "",
    max_tokens: null, rag_collection: null, repetition_penalty: null, repetition_detection: true,
  });
  openAccordions.add(modelsList.length - 1);
  markDirty();
  renderModels();
});

// --- Zustand -> Request-Body ------------------------------------------------
function buildPayload() {
  const num = (id) => { const v = $(id).value; return v === "" ? null : Number(v); };
  // default_serve_args-Keys dürfen nicht explizit null sein (serve_args_for()
  // erwartet fehlenden Key = "nimm den eingebauten Default", nicht null) -
  // ein leeres Feld entfernt den Key also, statt ihn auf null zu setzen.
  const dsa = Object.assign({}, state.default_serve_args || {});
  const dsaGmu = num("f-dsa_gpu_memory_utilization");
  const dsaMml = num("f-dsa_max_model_len");
  if (dsaGmu === null) delete dsa.gpu_memory_utilization; else dsa.gpu_memory_utilization = dsaGmu;
  if (dsaMml === null) delete dsa.max_model_len; else dsa.max_model_len = dsaMml;
  // Fürs Backfill unvollständiger Pricing-Overrides: der aktuell im Formular
  // stehende globale Default (nicht Pricing()'s eingebauter Klassen-Default -
  // sonst würde ein Override von nur EINER Seite die andere Seite überraschend
  // auf 3.0/15.0 zurücksetzen statt auf den eigenen konfigurierten Default).
  const currentDefaultPricing = {
    input_per_mtok: num("f-pricing_input_per_mtok") ?? 3.0,
    output_per_mtok: num("f-pricing_output_per_mtok") ?? 15.0,
  };
  const models = {};
  for (const m of modelsList) {
    const name = (m.name || "").trim();
    if (!name) continue;
    const { name: _drop, ...rest } = m;
    if (rest.pricing) {
      rest.pricing = {
        input_per_mtok: rest.pricing.input_per_mtok ?? currentDefaultPricing.input_per_mtok,
        output_per_mtok: rest.pricing.output_per_mtok ?? currentDefaultPricing.output_per_mtok,
      };
    }
    models[name] = rest;
  }
  return {
    host: $("f-host").value,
    port: parseInt($("f-port").value, 10),
    engine_host: $("f-engine_host").value,
    engine_port: parseInt($("f-engine_port").value, 10),
    hf_home: $("f-hf_home").value,
    vllm_bin: $("f-vllm_bin").value || null,
    api_key: { enabled: $("f-api_key_enabled").checked, key: $("f-api_key_key").value },
    idle_timeout_seconds: num("f-idle_timeout_seconds"),
    max_concurrent_models: parseInt($("f-max_concurrent_models").value, 10),
    gpu_memory_ceiling: parseFloat($("f-gpu_memory_ceiling").value),
    default_model: $("f-default_model").value || null,
    auto_reload_last_model: $("f-auto_reload_last_model").checked,
    startup_timeout_seconds: parseInt($("f-startup_timeout_seconds").value, 10),
    default_serve_args: dsa,
    default_pricing: {
      input_per_mtok: parseFloat($("f-pricing_input_per_mtok").value),
      output_per_mtok: parseFloat($("f-pricing_output_per_mtok").value),
    },
    models,
    rag: {
      enabled: $("f-rag_enabled").checked,
      qdrant_host: $("f-rag_qdrant_host").value,
      qdrant_port: parseInt($("f-rag_qdrant_port").value, 10),
      embedding_model: $("f-rag_embedding_model").value || null,
      default_collection: $("f-rag_default_collection").value,
      chunk_size_chars: parseInt($("f-rag_chunk_size_chars").value, 10),
      chunk_overlap_chars: parseInt($("f-rag_chunk_overlap_chars").value, 10),
      auto_rag_top_k: parseInt($("f-rag_auto_rag_top_k").value, 10),
      auto_rag_min_score: parseFloat($("f-rag_auto_rag_min_score").value),
    },
  };
}

function formatValidationError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map(d => `${(d.loc || []).join(".")}: ${d.msg}`).join("\n");
  }
  return JSON.stringify(detail);
}

// --- Aktionen --------------------------------------------------------------
function findDuplicateModelName() {
  const seen = new Set();
  for (const m of modelsList) {
    const name = (m.name || "").trim();
    if (!name) continue;
    if (seen.has(name)) return name;
    seen.add(name);
  }
  return null;
}

async function doSave() {
  const statusEl = $("save-status");
  const dup = findDuplicateModelName();
  if (dup) {
    statusEl.className = "status-msg error";
    statusEl.textContent = t("cfg.status.duplicateModel", { model: dup });
    return false;
  }
  statusEl.className = "status-msg";
  statusEl.textContent = "…";
  try {
    const res = await fetch("/config", { method: "POST", headers: authHeaders(), body: JSON.stringify(buildPayload()) });
    const data = await res.json();
    if (!res.ok) throw new Error(formatValidationError(data.detail ?? data));
    clearDirty();
    let msg = t("cfg.status.saved", { backup: data.backup || "–" });
    if (data.restart_recommended) msg += " " + t("cfg.status.restartRequired");
    statusEl.className = "status-msg ok";
    statusEl.textContent = msg;
    await loadConfig();
    return true;
  } catch (e) {
    statusEl.className = "status-msg error";
    statusEl.textContent = t("cfg.status.saveFailed", { msg: e.message });
    return false;
  }
}
$("save-btn").addEventListener("click", doSave);

$("discard-btn").addEventListener("click", () => {
  if (dirty && !confirm(t("confirm.discardChanges"))) return;
  loadConfig();
});

$("save-restart-btn").addEventListener("click", async () => {
  if (!confirm(t("confirm.restartService"))) return;
  const ok = await doSave();
  if (!ok) return;
  const statusEl = $("save-status");
  statusEl.className = "status-msg";
  statusEl.textContent = t("cfg.status.restarting");
  try {
    const res = await fetch("/config/restart", { method: "POST", headers: authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "restart failed");
  } catch (e) {
    statusEl.className = "status-msg error";
    statusEl.textContent = t("cfg.status.saveFailed", { msg: e.message });
    return;
  }
  pollForRestart();
});

async function pollForRestart() {
  const statusEl = $("save-status");
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 1500));
    try {
      const res = await fetch("/health", { cache: "no-store" });
      if (res.ok) {
        statusEl.className = "status-msg ok";
        statusEl.textContent = t("cfg.status.backOnline");
        return;
      }
    } catch (e) { /* Dienst noch nicht wieder erreichbar - weiter pollen */ }
  }
  statusEl.className = "status-msg error";
  statusEl.textContent = t("cfg.status.restartTimeout");
}

// --- Backups ---------------------------------------------------------------
let backupsCache = [];
async function loadBackups() {
  const res = await fetch("/config/backups", { headers: authHeaders() });
  const data = await res.json();
  backupsCache = data.backups || [];
  renderBackups();
}
function renderBackups() {
  const box = $("backups-box");
  if (backupsCache.length === 0) {
    box.innerHTML = `<div class="empty">${t("cfg.status.noBackups")}</div>`;
    return;
  }
  box.innerHTML = `<div class="table-scroll"><table><thead><tr>
    <th>${t("th.time")}</th><th>${t("cfg.th.filename")}</th><th>${t("cfg.th.size")}</th><th>${t("th.action")}</th>
    </tr></thead><tbody>` + backupsCache.map(b => `<tr>
      <td class="mono">${new Date(b.modified_at * 1000).toLocaleString(currentLang === "de" ? "de-DE" : "en-US")}</td>
      <td class="mono">${esc(b.filename)}</td>
      <td class="mono">${(b.size / 1024).toFixed(1)} KB</td>
      <td><button class="btn b-restore" data-filename="${esc(b.filename)}">${t("cfg.action.restore")}</button></td>
    </tr>`).join("") + `</tbody></table></div>`;
  document.querySelectorAll(".b-restore").forEach(el => {
    el.addEventListener("click", async () => {
      const filename = el.dataset.filename;
      if (!confirm(t("confirm.restoreBackup", { name: filename }))) return;
      const statusEl = $("save-status");
      try {
        const res = await fetch("/config/restore", { method: "POST", headers: authHeaders(), body: JSON.stringify({ filename }) });
        const data = await res.json();
        if (!res.ok) throw new Error(formatValidationError(data.detail ?? data));
        statusEl.className = "status-msg ok";
        statusEl.textContent = t("cfg.status.restored");
        await loadConfig();
      } catch (e) {
        statusEl.className = "status-msg error";
        statusEl.textContent = t("cfg.status.saveFailed", { msg: e.message });
      }
    });
  });
}

window.addEventListener("beforeunload", (e) => {
  if (dirty) { e.preventDefault(); e.returnValue = ""; }
});

populateLangSelect();
applyStaticI18n();
loadConfig();
</script>
</body>
</html>
"""

CONFIG_DASHBOARD_HTML = CONFIG_DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
