"""Live-Status-Dashboard: /dashboard (HTML), /dashboard/status (JSON-Snapshot,
für curl/Debugging), /dashboard/ws (WebSocket, echtes Live-Push bei jeder
Änderung + 1s-Heartbeat für die reinen Engine-Metriken).

WebSocket statt SSE: SSE über StreamingResponse lief durch FastAPIs
BaseHTTPMiddleware (unser API-Key-Check), was zu abgehackten/verschwindenden
Updates im Browser führte. WebSocket-Verbindungen laufen nicht durch den
HTTP-Middleware-Stack, daher wird die Auth hier direkt in der Handshake-Message
geprüft."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from . import downloader, process_manager, system_metrics, telemetry
from .catalog import list_cached_models
from .config import get_config

router = APIRouter()

HEARTBEAT_SECONDS = 1.0


async def build_snapshot() -> dict:
    cfg = get_config()
    engine_metrics = {}
    if process_manager.engine.model:
        engine_metrics = await telemetry.fetch_engine_metrics()
    now = time.time()
    return {
        "server_time": now,
        "engine": process_manager.engine.status(),
        "default_model": cfg.default_model,
        "last_request_at": telemetry.last_request_at,
        "seconds_since_last_request": (
            round(now - telemetry.last_request_at, 1) if telemetry.last_request_at else None
        ),
        "active_requests": list(telemetry.active_requests.values()),
        "recent_requests": list(telemetry.recent_requests),
        "engine_metrics": engine_metrics,
        "downloads": [j for j in downloader.list_jobs() if j["state"] != "done"][:10],
        "model_history": _model_history_with_current(),
        "models_catalog": _models_catalog(cfg),
        "system_metrics": await system_metrics.fetch_system_metrics(),
    }


def _models_catalog(cfg) -> list[dict]:
    """Alle nutzbaren Modelle fürs Dashboard: registrierte (config.json) +
    zusätzlich lokal gecachte, aber nicht registrierte. Liefert die Felder, die
    das Klick-Modal für den HF-Link und den VS-Code-JSON-Schnipsel braucht."""
    cached = set(list_cached_models(cfg.hf_home))
    default_mml = (cfg.default_serve_args or {}).get("max_model_len", 32768)
    out = []
    for name, mcfg in cfg.models.items():
        out.append({
            "model": name,
            "cached": name in cached,
            "loaded": process_manager.engine.model == name,
            "enabled": mcfg.enabled,
            "vision": mcfg.vision,
            "tool_calling": mcfg.enable_auto_tool_choice,
            "max_model_len": cfg.serve_args_for(name)[1],
            "notes": mcfg.notes,
        })
    known = {m["model"] for m in out}
    for name in sorted(cached - known):
        out.append({
            "model": name,
            "cached": True,
            "loaded": process_manager.engine.model == name,
            "enabled": True,
            "vision": False,
            "tool_calling": False,
            "max_model_len": default_mml,
            "notes": "Lokal gecacht, nicht in config.json registriert.",
        })
    return sorted(out, key=lambda m: m["model"].lower())


def _model_history_with_current() -> list[dict]:
    """Verlauf wie 'ollama ps', aber historisch: aktuelle Session (falls
    vorhanden) zuerst, dann vergangene Sessions (Wechsel/Entladen/Abstürze)."""
    history = list(process_manager.model_history)[:19]
    eng = process_manager.engine
    if eng.model is not None and eng.state in ("loading", "ready"):
        current = {
            "model": eng.model,
            "loaded_at": eng.started_at,
            "unloaded_at": None,
            "duration_seconds": None,
            "reason": eng.state,  # "loading" oder "ready" markiert die laufende Session
        }
        return [current] + history
    return history


@router.get("/dashboard/status")
async def dashboard_status():
    return await build_snapshot()


@router.websocket("/dashboard/ws")
async def dashboard_ws(websocket: WebSocket):
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

    q = telemetry.subscribe()
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
        telemetry.unsubscribe(q)


@router.get("/dashboard")
async def dashboard_page():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Manager – Live Status</title>
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
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; }
  .sub { color: var(--text-dim); font-size: 13px; margin-bottom: 20px; }
  .conn { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--text-dim); }
  .dot { width:8px; height:8px; border-radius:50%; background: var(--bad); }
  .dot.live { background: var(--good); box-shadow: 0 0 6px var(--good); }
  #theme-toggle {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; width:36px; height:36px; font-size:16px; cursor:pointer; flex:0 0 auto;
  }
  #theme-toggle:hover { background:var(--panel-2); }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:14px; margin-bottom: 20px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card .label { color:var(--text-dim); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
  .card .value { font-size:22px; font-weight:600; }
  .card .value.small { font-size:15px; font-family:var(--mono); }
  .card .hint { color:var(--text-dim); font-size:12px; margin-top:6px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge.ok { background: var(--good-bg); color: var(--good); }
  .badge.error { background: var(--bad-bg); color: var(--bad); }
  .badge.running { background: var(--accent-bg); color: var(--accent); }
  .badge.idle { background: var(--panel-2); color: var(--text-dim); }
  .badge.loading { background: var(--warn-bg); color: var(--warn); animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.5; } }
  section { margin-bottom: 26px; }
  section h2 { font-size: 14px; color: var(--text-dim); text-transform:uppercase; letter-spacing:.04em; margin: 0 0 10px; }
  table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  td.mono, th.mono { font-family: var(--mono); }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .bar-bg { background:var(--panel-2); border-radius:6px; height:6px; overflow:hidden; margin-top:6px; }
  .bar-fg { background:var(--accent); height:100%; transition: width .3s; }
  .active-req { border-left: 3px solid var(--accent); padding-left:10px; }

  .chart-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr)); gap:14px; }
  .chart-card canvas { width:100%; height:70px; display:block; margin-top:8px; }

  .model-grid { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
  @media (max-width: 640px) { .model-grid { grid-template-columns: 1fr; } }
  .model-item {
    background:var(--panel); border:1px solid var(--border); border-radius:10px;
    padding:12px 14px; cursor:pointer; transition: border-color .15s, background .15s;
  }
  .model-item:hover { border-color: var(--accent); background: var(--panel-2); }
  .model-item .name { font-family: var(--mono); font-size:13px; font-weight:600; word-break: break-all; }
  .model-item .badges { margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  .model-item .badges .badge { font-size:10px; }

  .modal-overlay {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.5);
    align-items:center; justify-content:center; z-index:100; padding:20px;
  }
  .modal-overlay.open { display:flex; }
  .modal {
    background:var(--panel); border:1px solid var(--border); border-radius:12px;
    max-width:640px; width:100%; max-height:85vh; overflow-y:auto; padding:22px;
  }
  .modal h3 { margin:0 0 4px; font-family: var(--mono); font-size:16px; word-break: break-all; }
  .modal .badges { display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 16px; }
  .modal a.hf-link {
    display:inline-flex; align-items:center; gap:6px; color:var(--accent);
    text-decoration:none; font-size:13px; margin-bottom:16px;
  }
  .modal a.hf-link:hover { text-decoration:underline; }
  .modal .json-label { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
  .modal .json-label span { font-size:12px; color:var(--text-dim); text-transform:uppercase; letter-spacing:.04em; }
  .modal button.copy-btn {
    background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer;
  }
  .modal button.copy-btn:hover { border-color: var(--accent); }
  .modal pre {
    background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
    padding:12px; font-family: var(--mono); font-size:12px; overflow-x:auto; margin:0;
  }
  .modal .close-btn {
    position:absolute; top:16px; right:20px; background:none; border:none;
    color:var(--text-dim); font-size:20px; cursor:pointer; line-height:1;
  }
  .modal { position:relative; }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>vLLM Manager – Live Status</h1>
      <div class="sub"><span class="conn"><span class="dot" id="conn-dot"></span><span id="conn-text">verbinde…</span></span></div>
    </div>
    <button id="theme-toggle" title="Theme wechseln">🌙</button>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">Geladenes Modell</div>
      <div class="value small" id="loaded-model">– <span class="badge idle" id="model-state-badge">idle</span></div>
      <div class="hint" id="model-uptime"></div>
    </div>
    <div class="card">
      <div class="label">Letzter Prompt</div>
      <div class="value" id="last-prompt">–</div>
      <div class="hint" id="last-prompt-abs"></div>
    </div>
    <div class="card">
      <div class="label">Requests laufend / wartend</div>
      <div class="value" id="running-waiting">–</div>
      <div class="hint">vLLM-Engine-Scheduler</div>
    </div>
    <div class="card">
      <div class="label">KV-Cache-Auslastung</div>
      <div class="value" id="kv-cache">–</div>
      <div class="bar-bg"><div class="bar-fg" id="kv-cache-bar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="label">Ø TTFT / Ø Token-Latenz</div>
      <div class="value small" id="avg-ttft">–</div>
      <div class="hint" id="avg-tpot"></div>
    </div>
    <div class="card">
      <div class="label">Tokens gesamt (Prompt / Generiert)</div>
      <div class="value small" id="token-totals">–</div>
    </div>
  </div>

  <section>
    <h2>Aktive Anfrage</h2>
    <div id="active-request-box"></div>
  </section>

  <section>
    <h2>System-Auslastung</h2>
    <div class="chart-grid">
      <div class="card chart-card">
        <div class="label">GPU-Auslastung</div>
        <div class="value" id="gpu-percent">–</div>
        <div class="hint" id="gpu-extra"></div>
        <canvas id="gpu-chart" width="400" height="70"></canvas>
      </div>
      <div class="card chart-card">
        <div class="label">RAM-Auslastung (Unified Memory)</div>
        <div class="value" id="ram-percent">–</div>
        <div class="hint" id="ram-extra"></div>
        <canvas id="ram-chart" width="400" height="70"></canvas>
      </div>
    </div>
  </section>

  <section>
    <h2>Modell-Verlauf</h2>
    <div id="history-box"></div>
  </section>

  <section>
    <h2>Letzte Anfragen</h2>
    <div id="recent-box"></div>
  </section>

  <section>
    <h2>Verfügbare Modelle</h2>
    <div class="model-grid" id="models-catalog-box"></div>
  </section>

  <section>
    <h2>Downloads (laufend)</h2>
    <div id="downloads-box"></div>
  </section>

  <div class="modal-overlay" id="model-modal-overlay">
    <div class="modal">
      <button class="close-btn" id="modal-close">✕</button>
      <h3 id="modal-title">–</h3>
      <div class="badges" id="modal-badges"></div>
      <a class="hf-link" id="modal-hf-link" href="#" target="_blank" rel="noopener">🤗 Auf HuggingFace ansehen ↗</a>
      <div class="json-label">
        <span>VS Code Custom-Endpoint (chatLanguageModels.json)</span>
        <button class="copy-btn" id="modal-copy-btn">Kopieren</button>
      </div>
      <pre id="modal-json"></pre>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);

function fmtAgo(seconds) {
  if (seconds === null || seconds === undefined) return "noch keine";
  if (seconds < 2) return "gerade eben";
  if (seconds < 60) return Math.floor(seconds) + "s her";
  if (seconds < 3600) return Math.floor(seconds/60) + "m her";
  return Math.floor(seconds/3600) + "h her";
}
function fmtMs(ms) { return (ms === null || ms === undefined) ? "–" : (ms < 1000 ? Math.round(ms) + " ms" : (ms/1000).toFixed(2) + " s"); }
function fmtPct(x) { return (x === null || x === undefined) ? "–" : Math.round(x*100) + "%"; }
// Menschenlesbare Dauer, z.B. 3000s -> "50m", 65s -> "1m 5s", 45s -> "45s"
function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  seconds = Math.round(seconds);
  if (seconds < 60) return seconds + "s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}
function esc(s) { return (s ?? "").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function shortModel(m) { return m ? m.split("/").pop() : "–"; }
function reasonLabel(r) {
  if (r === "ready") return "aktuell geladen";
  if (r === "loading") return "lädt gerade (Kaltstart)";
  if (r === "manual_unload") return "manuell entladen";
  if (r === "idle_timeout") return "Idle-Timeout";
  if (r === "crashed") return "abgestürzt";
  if (r === "timeout") return "Start-Timeout";
  if (r === "shutdown") return "Dienst-Neustart";
  if (r && r.startsWith("replaced_by:")) return "ersetzt durch " + esc(shortModel(r.slice(13)));
  return esc(r || "–");
}
function reasonBadgeClass(r) {
  if (r === "ready") return "ok";
  if (r === "loading") return "loading";
  if (r === "crashed" || r === "timeout") return "error";
  return "idle";
}

// --- Live-Charts (GPU/RAM) ----------------------------------------------
// Reine Canvas-Sparklines ohne externe Lib: rollierender Verlauf im Browser,
// gefüttert von jedem WS-Snapshot (~1x/s durch den Heartbeat).
const MAX_POINTS = 120;
const gpuHistory = [];
const ramHistory = [];

function pushHistory(arr, val) {
  arr.push(typeof val === "number" ? val : null);
  if (arr.length > MAX_POINTS) arr.shift();
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function drawChart(canvas, history, varName) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const pts = history.filter(v => v !== null).length;
  if (pts < 2) return;
  const color = cssVar(varName) || "#888";
  const step = w / (MAX_POINTS - 1);
  const offset = MAX_POINTS - history.length;
  ctx.beginPath();
  let started = false;
  history.forEach((v, i) => {
    if (v === null || v === undefined) return;
    const x = (offset + i) * step;
    const y = h - Math.max(0, Math.min(1, v)) * (h - 4) - 2;
    if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.lineTo((offset + history.length - 1) * step, h);
  ctx.lineTo(offset * step, h);
  ctx.closePath();
  ctx.globalAlpha = 0.15;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function render(data) {
  const eng = data.engine || {};
  const modelNameEl = $("loaded-model");
  const badge = $("model-state-badge");
  modelNameEl.childNodes[0].nodeValue = (eng.loaded_model ? shortModel(eng.loaded_model) : "kein Modell geladen") + " ";
  if (eng.state === "loading") {
    badge.className = "badge loading"; badge.textContent = "🥶 Kaltstart läuft…";
  } else if (eng.state === "ready") {
    badge.className = "badge ok"; badge.textContent = "✅ bereit";
  } else {
    badge.className = "badge idle"; badge.textContent = "idle";
  }
  $("model-uptime").textContent = eng.running
    ? (eng.state === "loading" ? "lädt seit " : "läuft seit ") + fmtDuration(eng.uptime_seconds)
    : (eng.last_error ? "Letzter Fehler: " + esc(eng.last_error) : "");

  $("last-prompt").textContent = fmtAgo(data.seconds_since_last_request);
  $("last-prompt-abs").textContent = data.last_request_at
    ? new Date(data.last_request_at*1000).toLocaleString("de-DE") : "";

  const m = data.engine_metrics || {};
  $("running-waiting").textContent = (m.num_requests_running ?? "–") + " / " + (m.num_requests_waiting ?? "–");
  $("kv-cache").textContent = fmtPct(m.kv_cache_usage_perc);
  $("kv-cache-bar").style.width = (m.kv_cache_usage_perc ? Math.round(m.kv_cache_usage_perc*100) : 0) + "%";
  $("avg-ttft").textContent = "TTFT: " + fmtMs(m.avg_ttft_ms);
  $("avg-tpot").textContent = "Ø ms/Token: " + fmtMs(m.avg_tpot_ms) + (m.avg_tpot_ms ? "  (~" + Math.round(1000/m.avg_tpot_ms) + " Tok/s)" : "");
  $("token-totals").textContent = (m.prompt_tokens_total ?? "–") + " / " + (m.generation_tokens_total ?? "–");

  const sys = data.system_metrics || {};
  $("gpu-percent").textContent = fmtPct(sys.gpu_percent);
  $("gpu-extra").textContent = [
    sys.gpu_temp_c != null ? sys.gpu_temp_c + " °C" : null,
    sys.gpu_power_w != null ? Math.round(sys.gpu_power_w) + " W" : null,
  ].filter(Boolean).join(" · ");
  pushHistory(gpuHistory, sys.gpu_percent);
  drawChart($("gpu-chart"), gpuHistory, "--accent");

  $("ram-percent").textContent = fmtPct(sys.ram_percent);
  $("ram-extra").textContent = sys.ram_used_gb != null ? `${sys.ram_used_gb} / ${sys.ram_total_gb} GB` : "";
  pushHistory(ramHistory, sys.ram_percent);
  drawChart($("ram-chart"), ramHistory, "--good");

  const active = data.active_requests || [];
  if (active.length === 0) {
    $("active-request-box").innerHTML = '<div class="empty">Keine aktive Anfrage</div>';
  } else {
    $("active-request-box").innerHTML = active.map(r => {
      const elapsed = (Date.now()/1000 - r.started_at);
      const loadHint = r.queued_ms ? ` · Modell-Ladezeit: ${fmtMs(r.queued_ms)}` : "";
      return `<div class="card active-req">
        <div class="value small">${esc(shortModel(r.model))} <span class="badge running">läuft</span></div>
        <div class="hint">seit ${fmtDuration(elapsed)}${loadHint} · TTFT: ${fmtMs(r.ttft_ms)} · ~${r.tokens_streamed} Tokens gestreamt (approx.)</div>
      </div>`;
    }).join("");
  }

  const hist = data.model_history || [];
  if (hist.length === 0) {
    $("history-box").innerHTML = '<div class="empty">Noch kein Modell geladen seit Dienststart</div>';
  } else {
    $("history-box").innerHTML = `<table><thead><tr>
      <th>Modell</th><th>Status</th><th>Geladen um</th><th>Entladen um</th><th>Dauer</th>
      </tr></thead><tbody>` + hist.map(h => {
        const ongoing = h.unloaded_at === null || h.unloaded_at === undefined;
        const duration = ongoing
          ? fmtDuration(h.loaded_at ? (Date.now()/1000 - h.loaded_at) : null)
          : fmtDuration(h.duration_seconds);
        return `<tr>
        <td>${esc(shortModel(h.model))}</td>
        <td><span class="badge ${reasonBadgeClass(h.reason)}">${reasonLabel(h.reason)}</span></td>
        <td class="mono">${h.loaded_at ? new Date(h.loaded_at*1000).toLocaleTimeString("de-DE") : "–"}</td>
        <td class="mono">${ongoing ? "–" : new Date(h.unloaded_at*1000).toLocaleTimeString("de-DE")}</td>
        <td class="mono">${duration}</td>
      </tr>`;
      }).join("") + `</tbody></table>`;
  }

  const dls = data.downloads || [];
  if (dls.length === 0) {
    $("downloads-box").innerHTML = '<div class="empty">Keine laufenden Downloads</div>';
  } else {
    $("downloads-box").innerHTML = `<table><thead><tr>
      <th>Modell</th><th>Status</th><th>Fortschritt</th><th>Geschwindigkeit</th><th>ETA</th>
      </tr></thead><tbody>` + dls.map(j => `<tr>
        <td>${esc(j.model)}</td>
        <td>${esc(j.state)}</td>
        <td class="mono">${Math.min(j.percent,100)}% (${(Math.min(j.bytes_done,j.bytes_total)/1e9).toFixed(1)}/${(j.bytes_total/1e9).toFixed(1)} GB)</td>
        <td class="mono">${j.speed_mbps} MB/s</td>
        <td class="mono">${fmtDuration(j.eta_seconds)}</td>
      </tr>`).join("") + `</tbody></table>`;
  }

  const recent = data.recent_requests || [];
  if (recent.length === 0) {
    $("recent-box").innerHTML = '<div class="empty">Noch keine Anfragen seit Dienststart</div>';
  } else {
    $("recent-box").innerHTML = `<table><thead><tr>
      <th>Zeit</th><th>Modell</th><th>Status</th><th>Ladezeit</th><th>Dauer</th><th>TTFT</th><th>Prompt-Tok.</th><th>Compl.-Tok.</th>
      </tr></thead><tbody>` + recent.map(r => `<tr>
        <td class="mono">${new Date(r.started_at*1000).toLocaleTimeString("de-DE")}</td>
        <td>${esc(shortModel(r.model))}</td>
        <td><span class="badge ${r.status === 'ok' ? 'ok' : 'error'}">${esc(r.status)}</span></td>
        <td class="mono">${r.queued_ms ? fmtMs(r.queued_ms) : "–"}</td>
        <td class="mono">${fmtMs(r.duration_ms)}</td>
        <td class="mono">${fmtMs(r.ttft_ms)}</td>
        <td class="mono">${r.prompt_tokens ?? "–"}</td>
        <td class="mono">${r.completion_tokens ?? "–"}</td>
      </tr>`).join("") + `</tbody></table>`;
  }

  latestCatalog = data.models_catalog || [];
  const catalog = latestCatalog;
  if (catalog.length === 0) {
    $("models-catalog-box").innerHTML = '<div class="empty">Keine Modelle bekannt</div>';
  } else {
    $("models-catalog-box").innerHTML = catalog.map((m, i) => `
      <div class="model-item" data-idx="${i}">
        <div class="name">${esc(shortModel(m.model))}</div>
        <div class="badges">
          ${m.loaded ? '<span class="badge running">geladen</span>' : ""}
          <span class="badge ${m.cached ? 'ok' : 'idle'}">${m.cached ? 'gecacht' : 'nicht gecacht'}</span>
          ${!m.enabled ? '<span class="badge error">deaktiviert</span>' : ""}
          ${m.vision ? '<span class="badge idle">Vision</span>' : ""}
        </div>
      </div>`).join("");
    document.querySelectorAll("#models-catalog-box .model-item").forEach(el => {
      el.addEventListener("click", () => openModal(latestCatalog[parseInt(el.dataset.idx, 10)]));
    });
  }
}

// --- Modell-Katalog / Klick-Modal ---------------------------------------
let latestCatalog = [];

function openModal(m) {
  $("modal-title").textContent = m.model;
  $("modal-hf-link").href = "https://huggingface.co/" + m.model;
  $("modal-badges").innerHTML = [
    m.loaded ? '<span class="badge running">geladen</span>' : "",
    `<span class="badge ${m.cached ? 'ok' : 'idle'}">${m.cached ? 'gecacht' : 'nicht gecacht'}</span>`,
    !m.enabled ? '<span class="badge error">deaktiviert</span>' : '<span class="badge ok">aktiviert</span>',
    m.vision ? '<span class="badge idle">Vision</span>' : "",
    m.tool_calling ? '<span class="badge idle">Tool-Calling</span>' : "",
  ].filter(Boolean).join("");

  const url = `${location.protocol}//${location.host}/v1`;
  const entry = {
    id: m.model,
    name: shortModel(m.model),
    url: url,
    toolCalling: !!m.tool_calling,
    vision: !!m.vision,
    maxInputTokens: m.max_model_len,
    maxOutputTokens: m.max_model_len,
  };
  $("modal-json").textContent = JSON.stringify(entry, null, 2);

  $("model-modal-overlay").classList.add("open");
}

function closeModal() { $("model-modal-overlay").classList.remove("open"); }

$("modal-close").addEventListener("click", closeModal);
$("model-modal-overlay").addEventListener("click", (e) => {
  if (e.target.id === "model-modal-overlay") closeModal();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
$("modal-copy-btn").addEventListener("click", () => {
  navigator.clipboard.writeText($("modal-json").textContent).then(() => {
    const btn = $("modal-copy-btn");
    const old = btn.textContent;
    btn.textContent = "Kopiert ✓";
    setTimeout(() => { btn.textContent = old; }, 1500);
  });
});

// --- Theme-Toggle -----------------------------------------------------
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

// --- WebSocket-Verbindung ----------------------------------------------
let apiKey = sessionStorage.getItem("vllm_dashboard_key") || "";

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/dashboard/ws");

  ws.onopen = () => {
    ws.send(JSON.stringify({ api_key: apiKey }));
    $("conn-dot").classList.add("live");
    $("conn-text").textContent = "live";
  };

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    if (data.type === "auth_error") {
      const key = prompt("API-Key erforderlich (siehe config.json):");
      if (key) {
        apiKey = key;
        sessionStorage.setItem("vllm_dashboard_key", key);
      }
      ws.close();
      return;
    }
    render(data);
  };

  ws.onclose = () => {
    $("conn-dot").classList.remove("live");
    $("conn-text").textContent = "verbindung unterbrochen, versuche erneut…";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

connect();
</script>
</body>
</html>
"""
