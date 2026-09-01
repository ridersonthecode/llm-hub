"""Kleine, eigenständige Chat-Seite: /dashboard/chat.

BEWUSST NICHT in der Hauptnavigation verlinkt (kein Eintrag in dashboard.py -
siehe dortigen Topbar-Link-Block) - nur über die direkte URL erreichbar, siehe
Chat vom 2026-08-28 ("erstelle mir eine Seite, die nicht in der Navigation
gelistet ist"). Trotzdem regulär durch die ApiKeyMiddleware geschützt wie
jeder andere Endpoint (nur die HTML-Seite selbst ist exempt, siehe auth.py -
gleiches Muster wie /dashboard/config etc.).

Kein eigener Server-Zustand nötig: die Seite spricht direkt den normalen
OpenAI-kompatiblen Proxy an (POST /v1/chat/completions, stream:true) - exakt
denselben Pfad wie jeder externe Client (VS Code, curl, ...). Modell-Auswahl
kommt aus /dashboard/status (models_catalog), das ohnehin schon für den
Haupt-Dashboard existiert und keine Secrets enthält (anders als GET /config).

Reasoning (delta.reasoning, bei manchen Server-Versionen noch delta.
reasoning_content - live per curl gegen die installierte vLLM 0.26.0 geprüft,
2026-08-28, siehe JS-Kommentar bei der SSE-Verarbeitung) und normaler Text
(delta.content) werden getrennt behandelt: reasoning landet in einem eigenen,
einklappbaren "Denkprozess"-Block. Code-Blöcke
(```lang ... ```) werden client-seitig per Mini-Markdown-Parser (keine externe
Bibliothek/kein CDN - siehe Konvention in cost_dashboard.py: DataTables liegt
lokal unter /static/vendor/, nicht auf einem CDN, damit die Seite auch ohne
Internetzugriff des Browsers funktioniert) als eigener, monospaced Block mit
Sprache + Kopieren-Button dargestellt."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .dashboard import _LANGUAGES_JS

router = APIRouter()


@router.get("/dashboard/chat")
async def chat_dashboard_page():
    return HTMLResponse(CHAT_DASHBOARD_HTML)


CHAT_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Hub – Chat</title>
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
  html, body { height:100%; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    display:flex; flex-direction:column; padding: 16px 20px 20px;
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  a { color: var(--accent); }
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom: 12px; flex:0 0 auto; }
  .topbar-actions { display:flex; gap:8px; align-items:center; flex:0 0 auto; flex-wrap:wrap; }
  .sub { color: var(--text-dim); font-size: 13px; }
  select, #theme-toggle, .btn {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; height:36px; padding:0 12px; font-size:13px; cursor:pointer;
  }
  select { min-width: 220px; }
  #theme-toggle { width:36px; padding:0; font-size:16px; }
  #theme-toggle:hover, select:hover, .btn:hover { background:var(--panel-2); }
  .btn:disabled { opacity:.5; cursor:default; }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:var(--bubble-user-text); }
  .btn.primary:hover:not(:disabled) { filter:brightness(1.08); }
  .btn.danger:hover:not(:disabled) { border-color: var(--bad); color: var(--bad); }
  .btn.ghost { background:transparent; }

  .chat-shell {
    flex: 1 1 auto; min-height:0; display:flex; flex-direction:column;
    max-width: 900px; width:100%; margin: 0 auto; gap:10px;
  }
  .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; flex:0 0 auto; }
  .toolbar .spacer { flex:1; }
  .toolbar label { font-size:12px; color:var(--text-dim); display:flex; align-items:center; gap:6px; }
  #max-tokens { width:80px; background:var(--panel); border:1px solid var(--border); color:var(--text); border-radius:8px; height:32px; padding:0 8px; font-size:13px; }

  .messages-wrap { position:relative; flex:1 1 auto; min-height:0; display:flex; }
  #messages {
    flex:1 1 auto; min-height:0; overflow-y:auto; background:var(--panel);
    border:1px solid var(--border); border-radius:12px; padding: 16px;
    display:flex; flex-direction:column; gap:14px;
  }
  .empty-hint { color:var(--text-dim); font-size:13px; text-align:center; margin:auto; }

  /* Erscheint nur, wenn während des Streamens hochgescrollt wurde (siehe
     JS: stickToBottom) - liegt bewusst AUSSERHALB von #messages (eigenes
     Geschwisterelement in .messages-wrap), damit es beim Neuaufbau von
     #messages.innerHTML bei jedem Chunk nicht mit weggerissen wird und der
     Klick-Listener nur einmal gebunden werden muss. */
  .scroll-bottom-btn {
    position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
    width:36px; height:36px; border-radius:50%; z-index:2;
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    box-shadow:0 2px 10px rgba(0,0,0,.18); cursor:pointer; font-size:15px;
    display:none; align-items:center; justify-content:center;
  }
  .scroll-bottom-btn:hover { background:var(--panel-2); }

  .msg-row { display:flex; }
  .msg-row.user { justify-content:flex-end; }
  .msg-row.assistant { justify-content:flex-start; }
  .bubble {
    max-width: 85%; border-radius:14px; padding:10px 14px; font-size:14px; line-height:1.55;
    overflow-wrap:anywhere;
  }
  .msg-row.user .bubble { background:var(--bubble-user); color:var(--bubble-user-text); border-bottom-right-radius:4px; white-space:pre-wrap; }
  .msg-row.assistant .bubble { background:var(--bubble-assistant); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .msg-row.assistant .bubble.error { background:var(--bad-bg); border-color:var(--bad); color:var(--bad); }
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
  .reasoning.live summary { color:var(--accent); }
  .reasoning.live summary::after { content:""; width:6px; height:6px; border-radius:50%; background:var(--accent); animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.25;} }

  .code-block { margin: 8px 0; border-radius:8px; overflow:hidden; border:1px solid var(--border); }
  .code-block-bar {
    background:var(--code-bar); color:#9da5b4; font-family: var(--mono); font-size:11px;
    padding:5px 10px; display:flex; align-items:center; justify-content:space-between;
  }
  .code-copy-btn { background:transparent; border:1px solid #30363d; color:#9da5b4; border-radius:5px; padding:2px 8px; font-size:11px; cursor:pointer; }
  .code-copy-btn:hover { background:#30363d; color:#fff; }
  .code-block pre { margin:0; background:var(--code-bg); color:var(--code-text); padding:12px; overflow-x:auto; }
  .code-block code { font-family: var(--mono); font-size:12.5px; background:none; padding:0; }

  .status-line { font-size:12px; color:var(--text-dim); min-height:16px; flex:0 0 auto; }
  .status-line.err { color:var(--bad); }

  .composer { display:flex; gap:10px; align-items:flex-end; flex:0 0 auto; }
  #input-box {
    flex:1; resize:none; min-height:44px; max-height:200px; border-radius:12px;
    border:1px solid var(--border); background:var(--panel); color:var(--text);
    padding:11px 14px; font-size:14px; font-family:inherit; line-height:1.4;
  }
  #input-box:focus { outline:none; border-color:var(--accent); }
  .composer .btn { height:44px; min-width:44px; padding:0 16px; }

  .cursor { display:inline-block; width:7px; height:1em; background:var(--text-dim); vertical-align:text-bottom; animation:blink 1s step-start infinite; margin-left:1px; }
  @keyframes blink { 50% { opacity:0; } }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1 data-i18n="chat.title">Chat</h1>
      <div class="sub"><a href="/dashboard" data-i18n="nav.dashboardLink">← Dashboard</a></div>
    </div>
    <div class="topbar-actions">
      <select id="lang-select" data-i18n-title="lang.selectTitle" title="Language"></select>
      <button id="theme-toggle" data-i18n-title="theme.toggleTitle" title="Toggle theme">🌙</button>
    </div>
  </div>

  <div class="chat-shell">
    <div class="toolbar">
      <select id="model-select"></select>
      <button class="btn ghost" id="refresh-models-btn" data-i18n-title="chat.action.refreshModels" title="Refresh model list">🔄</button>
      <div class="spacer"></div>
      <label><span data-i18n="chat.label.maxTokens">Max tokens</span> <input type="number" id="max-tokens" min="1" placeholder="∞"></label>
      <button class="btn danger" id="clear-btn" data-i18n="chat.action.clear">Clear chat</button>
    </div>

    <div class="messages-wrap">
      <div id="messages"><div class="empty-hint" id="empty-hint" data-i18n="chat.hint.empty">Pick a model above and send a message to start.</div></div>
      <button class="scroll-bottom-btn" id="scroll-bottom-btn" data-i18n-title="chat.action.scrollToBottom" title="Scroll to bottom">⬇</button>
    </div>

    <div class="status-line" id="status-line"></div>

    <div class="composer">
      <textarea id="input-box" rows="1" data-i18n-placeholder="chat.placeholder.message" placeholder="Type a message… (Shift+Enter for a new line)"></textarea>
      <button class="btn primary" id="send-btn" data-i18n="chat.action.send">Send</button>
      <button class="btn danger" id="stop-btn" data-i18n="chat.action.stop" style="display:none;">Stop</button>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);

// --- i18n --------------------------------------------------------------
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
  document.title = "LLM Hub – " + t("chat.title");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
}
$("lang-select").addEventListener("change", (e) => {
  currentLang = e.target.value;
  localStorage.setItem("vllm_dashboard_lang", currentLang);
  applyStaticI18n();
  populateModelSelect();
});

// --- Theme ---------------------------------------------------------------
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

// --- API-Key (falls in config.json aktiviert, gleicher Key wie die anderen
// Dashboard-Seiten - sessionStorage wird zwischen ihnen geteilt, da alle vom
// selben Origin laufen) --------------------------------------------------
let apiKey = sessionStorage.getItem("vllm_dashboard_key") || "";
function authHeaders(extra) {
  const h = Object.assign({ "Content-Type": "application/json" }, extra || {});
  if (apiKey) h["Authorization"] = "Bearer " + apiKey;
  return h;
}
async function ensureApiKey() {
  const key = prompt(t("auth.apiKeyPrompt"));
  if (key) { apiKey = key; sessionStorage.setItem("vllm_dashboard_key", key); }
}

// navigator.clipboard gibt es nur in "sicheren Kontexten" (HTTPS/localhost) -
// dieses Dashboard läuft absichtlich über eine reine HTTP-LAN-IP, siehe
// dashboard.py copyText()-Kommentar. Gleicher Fallback hier für den
// "Kopieren"-Button an Code-Blöcken.
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve, reject) => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error("execCommand('copy') failed"));
    } catch (e) {
      document.body.removeChild(ta);
      reject(e);
    }
  });
}

// --- Mini-Markdown-Renderer -------------------------------------------
// Bewusst kein CDN/keine externe Bibliothek (siehe Moduldocstring) - deckt
// nur ab, was LLM-Antworten typischerweise nutzen: Fenced Code-Blöcke
// (inkl. eines noch offenen am Streaming-Ende), Inline-Code, **fett**/*kursiv*,
// Überschriften, einfache Listen, Links. Kein vollständiger CommonMark-Parser.
function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderInline(text) {
  const codes = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(`<code>${escapeHtml(code)}</code>`);
    return `\u0000${codes.length - 1}\u0000`;
  });
  text = escapeHtml(text);
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  text = text.replace(/\u0000(\d+)\u0000/g, (_, i) => codes[Number(i)]);
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
  const flushPara = () => {
    if (paraBuf.length) { htmlParts.push(`<p>${renderInline(paraBuf.join(" "))}</p>`); paraBuf = []; }
  };
  for (const line of lines) {
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    if (line.trim() === "") { flushPara(); flushList(); continue; }
    if (heading) {
      flushPara(); flushList();
      const level = heading[1].length + 2; // h3..h6 - h1/h2 bleiben der Seite selbst vorbehalten
      htmlParts.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
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
  // Text an jedem ``` aufsplitten: gerade Indizes = normaler Text, ungerade =
  // Code (Sprache in der ersten Zeile). Bei einer ungeraden Gesamtzahl von ```
  // (Streaming mitten in einem Codeblock) ist das letzte Segment ein noch
  // offener Codeblock - wird genauso gerendert, nur ohne abschließendes ```.
  const parts = text.split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      html += renderMarkdownBlock(parts[i]);
    } else {
      const seg = parts[i];
      const nl = seg.indexOf("\n");
      const lang = nl === -1 ? seg.trim() : seg.slice(0, nl).trim();
      const code = nl === -1 ? "" : seg.slice(nl + 1);
      html += renderCodeBlock(lang, code);
    }
  }
  return html;
}

// Delegiert auf #messages statt pro Code-Block einen eigenen Listener zu
// binden - Bubbles werden während des Streamens laufend neu gerendert
// (innerHTML ersetzt), einzeln gebundene Listener würden dabei verloren gehen.
$("messages").addEventListener("click", async (e) => {
  const btn = e.target.closest(".code-copy-btn");
  if (!btn) return;
  const codeEl = $(btn.dataset.target);
  if (!codeEl) return;
  try {
    await copyText(codeEl.textContent);
    const original = btn.textContent;
    btn.textContent = t("chat.action.copied");
    setTimeout(() => { btn.textContent = original; }, 1500);
  } catch (err) { /* still, kein Copy-Fallback mehr übrig - egal, nicht kritisch */ }
});

// --- Modell-Auswahl --------------------------------------------------------
// /dashboard/status liefert models_catalog (Name, task, enabled, ...) ohne
// jegliche Secrets (anders als GET /config, das u.a. hf_token/api_key.key roh
// mitliefert - für eine reine Modellliste hier bewusst NICHT verwendet).
// Zuletzt gewähltes Modell pro Browser gemerkt (localStorage - wie Theme/
// Sprache, siehe oben) und beim nächsten Öffnen der Seite automatisch wieder
// vorausgewählt. NUR die Modellwahl, nicht der Chatverlauf selbst (siehe
// Chat vom 2026-08-28 - Verlaufs-Persistenz wurde bewusst verworfen).
const LAST_MODEL_STORAGE_KEY = "vllm_chat_last_model";
function saveLastModel(model) {
  try { localStorage.setItem(LAST_MODEL_STORAGE_KEY, model); }
  catch (e) { /* z.B. Privatmodus/voller Speicher - nur Komfortfunktion, kein Problem */ }
}
function loadLastModel() {
  try { return localStorage.getItem(LAST_MODEL_STORAGE_KEY) || ""; }
  catch (e) { return ""; }
}

async function populateModelSelect() {
  const sel = $("model-select");
  const prev = sel.value;
  try {
    const res = await fetch("/dashboard/status", { headers: authHeaders() });
    if (res.status === 401) { await ensureApiKey(); return populateModelSelect(); }
    const data = await res.json();
    const models = (data.models_catalog || []).filter(m => m.task !== "embed" && m.enabled !== false);
    if (models.length === 0) {
      sel.innerHTML = `<option value="">${esc(t("chat.hint.noModels"))}</option>`;
      return;
    }
    sel.innerHTML = models.map(m =>
      `<option value="${esc(m.model)}">${esc(m.model)}${m.loaded ? " ⚡" : ""}</option>`
    ).join("");
    // Reihenfolge: 1) schon in dieser Seite ausgewählt (z.B. vor einem
    // manuellen 🔄-Klick) - 2) beim allerersten Laden das zuletzt genutzte
    // Modell dieses Browsers - 3) das globale default_model aus config.json.
    const lastModel = loadLastModel();
    if (models.some(m => m.model === prev)) sel.value = prev;
    else if (lastModel && models.some(m => m.model === lastModel)) sel.value = lastModel;
    else if (data.default_model && models.some(m => m.model === data.default_model)) sel.value = data.default_model;
  } catch (e) {
    sel.innerHTML = `<option value="">${esc(t("chat.hint.noModels"))}</option>`;
  }
}
$("refresh-models-btn").addEventListener("click", populateModelSelect);
$("model-select").addEventListener("change", (e) => { if (e.target.value) saveLastModel(e.target.value); });

// --- Chat-Zustand ----------------------------------------------------------
// Nur im Speicher dieser Seite - kein Persistieren über einen Reload hinweg
// (bewusst einfach gehalten, siehe Anfrage "eine kleine Chat-Funktion").
// reasoning_content wird beim erneuten Senden NICHT mit an die API zurück-
// geschickt (nur content) - Modelle erwarten ihre eigene Historie ohne
// Denkprozess vergangener Runden, genau wie reguläre OpenAI-kompatible Clients
// das handhaben.
let conversation = [];      // [{role, content}]
let currentAbort = null;    // AbortController der laufenden Anfrage, für den Stop-Button

function setBusy(busy) {
  $("send-btn").style.display = busy ? "none" : "";
  $("stop-btn").style.display = busy ? "" : "none";
  $("input-box").disabled = busy;
  $("model-select").disabled = busy;
}
function setStatus(text, isError) {
  const el = $("status-line");
  el.textContent = text || "";
  el.classList.toggle("err", !!isError);
}

// Auto-Scroll standardmäßig an, schaltet sich selbst ab, sobald während des
// Streamens hochgescrollt wird (siehe #messages "scroll"-Listener unten) -
// erst ein Klick auf den Pfeil-Button ODER manuelles Zurückscrollen ganz nach
// unten aktiviert es wieder.
let stickToBottom = true;
const NEAR_BOTTOM_PX = 32; // Toleranz, damit "praktisch unten" auch als unten zählt

function isNearBottom() {
  const box = $("messages");
  return box.scrollHeight - box.scrollTop - box.clientHeight < NEAR_BOTTOM_PX;
}
function updateScrollButton() {
  $("scroll-bottom-btn").style.display = stickToBottom ? "none" : "flex";
}
$("messages").addEventListener("scroll", () => {
  const nearBottom = isNearBottom();
  if (nearBottom !== stickToBottom) {
    stickToBottom = nearBottom;
    updateScrollButton();
  }
});
$("scroll-bottom-btn").addEventListener("click", () => {
  stickToBottom = true;
  scrollToBottom();
  updateScrollButton();
});

function scrollToBottom() {
  // Nur tatsächlich scrollen, wenn stickToBottom aktiv ist (siehe oben) -
  // renderMessages() ruft das bei JEDEM Streaming-Chunk auf, ein hoch-
  // gescrollter Nutzer soll dabei nicht ständig wieder nach unten gerissen
  // werden.
  if (!stickToBottom) return;
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

function renderMessages() {
  const box = $("messages");
  $("empty-hint")?.remove();
  if (conversation.length === 0) {
    box.innerHTML = `<div class="empty-hint" id="empty-hint">${esc(t("chat.hint.empty"))}</div>`;
    return;
  }
  box.innerHTML = conversation.map((m, i) => {
    if (m.role === "user") {
      return `<div class="msg-row user"><div class="bubble">${escapeHtml(m.content)}</div></div>`;
    }
    const reasoningHtml = m.reasoning
      ? `<details class="reasoning ${m.streamingReasoning ? "live" : ""}" ${m.streamingReasoning || !m.content ? "open" : ""}>
           <summary>${esc(t("chat.label.reasoning"))}</summary>
           <div class="reasoning-body">${escapeHtml(m.reasoning)}</div>
         </details>`
      : "";
    const bodyHtml = m.error
      ? escapeHtml(m.content || m.error)
      : renderMarkdown(m.content || "") + (m.streaming && m.content ? '<span class="cursor"></span>' : "");
    const waitingHtml = (m.streaming && !m.content && !m.reasoning) ? `<span class="cursor"></span>` : "";
    return `<div class="msg-row assistant"><div class="bubble ${m.error ? "error" : ""}" data-idx="${i}">${reasoningHtml}${bodyHtml}${waitingHtml}</div></div>`;
  }).join("");
  scrollToBottom();
}

async function sendMessage() {
  const input = $("input-box");
  const text = input.value.trim();
  const model = $("model-select").value;
  if (!text || !model) return;

  conversation.push({ role: "user", content: text });
  const assistantMsg = { role: "assistant", content: "", reasoning: "", streaming: true, streamingReasoning: false };
  conversation.push(assistantMsg);
  input.value = "";
  input.style.height = "auto";
  // Eigene, aktiv gesendete Nachricht -> wieder ganz nach unten springen und
  // Auto-Scroll reaktivieren, auch wenn vorher hochgescrollt war (siehe
  // stickToBottom oben) - dasselbe Verhalten wie in den meisten Chat-Apps.
  stickToBottom = true;
  updateScrollButton();
  renderMessages();
  setBusy(true);
  setStatus(t("chat.status.waiting"));

  const controller = new AbortController();
  currentAbort = controller;

  const body = {
    model,
    stream: true,
    messages: conversation.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
  };
  const maxTokens = parseInt($("max-tokens").value, 10);
  if (maxTokens > 0) body.max_tokens = maxTokens;

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body), signal: controller.signal,
    });
    if (res.status === 401) {
      await ensureApiKey();
      conversation.splice(-2, 2);
      renderMessages();
      setBusy(false);
      currentAbort = null;
      input.value = text;
      return;
    }
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail || j.error || detail; } catch (e) { /* kein JSON-Body */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    setStatus("");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let firstToken = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop(); // letzte, evtl. unvollständige Zeile für die nächste Runde aufheben
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        if (!payload) continue;
        let evt;
        try { evt = JSON.parse(payload); } catch (e) { continue; }
        const delta = (evt.choices && evt.choices[0] && evt.choices[0].delta) || {};
        // Feldname für den Denkprozess je nach vLLM-Version unterschiedlich:
        // aktuelle Versionen (siehe installierte 0.26.0, chat_completion/
        // protocol.py) liefern "reasoning", ältere/andere Server evtl. noch
        // das inzwischen deprecated "reasoning_content" - live per curl
        // gegen den echten Stream geprüft (2026-08-28), main.py prüft an
        // anderer Stelle bisher NUR reasoning_content (siehe dortiger Fund).
        const reasoning = delta.reasoning || delta.reasoning_content;
        if (firstToken && (delta.content || reasoning)) {
          firstToken = false;
          setStatus("");
        }
        if (reasoning) {
          assistantMsg.reasoning += reasoning;
          assistantMsg.streamingReasoning = true;
        }
        if (delta.content) {
          assistantMsg.content += delta.content;
          assistantMsg.streamingReasoning = false;
        }
        if (delta.content || reasoning) renderMessages();
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      setStatus(t("chat.status.stopped"));
    } else {
      assistantMsg.error = true;
      assistantMsg.content = t("chat.status.streamError", { msg: e.message });
      setStatus(t("chat.status.streamError", { msg: e.message }), true);
    }
  } finally {
    assistantMsg.streaming = false;
    assistantMsg.streamingReasoning = false;
    renderMessages();
    setBusy(false);
    currentAbort = null;
  }
}

$("send-btn").addEventListener("click", sendMessage);
$("stop-btn").addEventListener("click", () => { if (currentAbort) currentAbort.abort(); });
$("input-box").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("input-box").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
});
$("clear-btn").addEventListener("click", () => {
  if (conversation.length === 0) return;
  if (!confirm(t("chat.confirm.clear"))) return;
  conversation = [];
  stickToBottom = true;
  updateScrollButton();
  renderMessages();
  setStatus("");
});

populateLangSelect();
applyStaticI18n();
updateScrollButton();
renderMessages();
populateModelSelect();
</script>
</body>
</html>
"""

CHAT_DASHBOARD_HTML = CHAT_DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
