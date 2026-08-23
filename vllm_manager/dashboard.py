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
import json
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from . import catalog, cost_tracker, downloader, process_manager, system_metrics, telemetry
from .catalog import list_cached_models
from .config import get_config

router = APIRouter()

HEARTBEAT_SECONDS = 1.0

# i18n: jede Datei in languages/ ist eine Sprache (Dateiname ohne .json = Code,
# z.B. "en.json" -> "en"). Neue Sprache hinzufügen = neue Datei ablegen, kein
# Code-Änderung nötig. Englisch ist die Standardsprache.
LANGUAGES_DIR = Path(__file__).resolve().parent / "languages"
DEFAULT_LANGUAGE = "en"


def _load_languages() -> dict[str, dict[str, str]]:
    langs: dict[str, dict[str, str]] = {}
    for f in sorted(LANGUAGES_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            langs[f.stem] = json.load(fh)
    return langs


LANGUAGES = _load_languages()
# Als JS-Objekt in die Seite eingebettet (kein Extra-Request nötig). "</" wird
# maskiert, damit ein Übersetzungstext nicht versehentlich das <script>-Tag
# beenden kann.
_LANGUAGES_JS = json.dumps(LANGUAGES, ensure_ascii=False).replace("</", "<\\/")


async def build_snapshot() -> dict:
    cfg = get_config()
    now = time.time()
    return {
        "server_time": now,
        "engines": await _engines_snapshot(),
        "pool": _pool_budget(cfg),
        "default_model": cfg.default_model,
        "last_request_at": telemetry.last_request_at,
        "seconds_since_last_request": (
            round(now - telemetry.last_request_at, 1) if telemetry.last_request_at else None
        ),
        "active_requests": _active_requests_with_cost(),
        "recent_requests": _recent_requests_with_cost(),
        "downloads": [j for j in downloader.list_jobs() if j["state"] != "done"][:10],
        "model_history": _model_history_with_current(),
        "models_catalog": await _models_catalog(cfg),
        "system_metrics": await system_metrics.fetch_system_metrics(),
    }


def _active_requests_with_cost() -> list[dict]:
    """Live-Schätzung fürs fiktive Kostentracking (siehe cost_tracker.py):
    NUR die Ausgabeseite, aus den approximativen Chunk-Zählern (tokens_streamed
    + reasoning_tokens_streamed) - die Prompt-Größe ist erst nach Requestende
    bekannt, daher bewusst nicht mitgerechnet (kein falscher Gesamtbetrag)."""
    out = []
    for r in telemetry.active_requests.values():
        d = dict(r)
        tokens_so_far = (r.get("tokens_streamed") or 0) + (r.get("reasoning_tokens_streamed") or 0)
        _, out_rate = cost_tracker.pricing_for(r["model"])
        d["estimated_output_cost_usd"] = round(tokens_so_far / 1_000_000 * out_rate, 6)
        out.append(d)
    return out


def _recent_requests_with_cost() -> list[dict]:
    """Exakte Kosten (nur wenn prompt_tokens/completion_tokens bekannt sind -
    siehe cost_tracker.compute_cost) für die letzten ~30 Anfragen. Wird bei
    jedem Snapshot live gegen die AKTUELLE Preiskonfiguration neu berechnet
    (anders als der persistente Datensatz auf /dashboard/costs, der den Preis
    zum Zeitpunkt der Anfrage festhält, siehe cost_tracker.record_request)."""
    out = []
    for r in telemetry.recent_requests:
        d = dict(r)
        cost = cost_tracker.compute_cost(r["model"], r.get("prompt_tokens"), r.get("completion_tokens"))
        d["cost_usd"] = cost["cost_usd"] if cost else None
        out.append(d)
    return out


async def _engines_snapshot() -> list[dict]:
    """Status + Live-Metriken jeder aktuell laufenden Engine im Hot Pool
    (bei max_concurrent_models=1 also höchstens eine)."""
    engs = list(process_manager.engines.values())

    async def _metrics(e: process_manager.EngineState) -> dict:
        return await telemetry.fetch_engine_metrics(e.port) if e.state == "ready" else {}

    metrics_list = await asyncio.gather(*[_metrics(e) for e in engs])
    out = []
    for eng, metrics in zip(engs, metrics_list):
        d = eng.status()
        d["metrics"] = metrics
        out.append(d)
    return out


def _pool_budget(cfg) -> dict:
    """GPU-Speicherbudget des Hot Pools: Summe der gpu_memory_utilization
    aller aktuell laufenden Engines gegen gpu_memory_ceiling."""
    used = sum(cfg.serve_args_for(e.model)[0] for e in process_manager.engines.values())
    return {
        "used": round(used, 3),
        "ceiling": cfg.gpu_memory_ceiling,
        "slots_used": len(process_manager.engines),
        "slots_total": cfg.max_concurrent_models,
    }


async def _models_catalog(cfg) -> list[dict]:
    """Alle nutzbaren Modelle fürs Dashboard: registrierte (config.json) +
    zusätzlich lokal gecachte, aber nicht registrierte. Liefert die Felder, die
    das Klick-Modal für den HF-Link und den VS-Code-JSON-Schnipsel braucht."""
    cached = set(await list_cached_models(cfg.hf_home))
    default_mml = (cfg.default_serve_args or {}).get("max_model_len", 32768)
    out = []
    default_gmu = (cfg.default_serve_args or {}).get("gpu_memory_utilization", 0.5)
    for name, mcfg in cfg.models.items():
        gmu, mml = cfg.serve_args_for(name)
        out.append({
            "model": name,
            "cached": name in cached,
            "loaded": process_manager.is_ready(name),
            "enabled": mcfg.enabled,
            "vision": mcfg.vision,
            "tool_calling": mcfg.enable_auto_tool_choice,
            "reasoning": bool(mcfg.reasoning_parser),
            "task": mcfg.task,
            "max_model_len": mml,
            "gpu_memory_utilization": gmu,
            "notes": mcfg.notes,
        })
    known = {m["model"] for m in out}
    for name in sorted(cached - known):
        out.append({
            "model": name,
            "cached": True,
            "loaded": process_manager.is_ready(name),
            "enabled": True,
            "vision": False,
            "tool_calling": False,
            "reasoning": False,
            "task": "generate",
            "max_model_len": default_mml,
            "gpu_memory_utilization": default_gmu,
            "notes": "Lokal gecacht, nicht in config.json registriert.",
        })
    # Speicherplatz je Modell (nur für tatsächlich gecachte - sonst wäre es
    # immer 0 statt "unbekannt"), parallel statt nacheinander abgefragt
    # (TTL-gecacht, siehe catalog.get_cached_size_bytes - sonst würde ein
    # os.walk() über hunderte GB bei jedem WebSocket-Heartbeat den
    # Event-Loop blockieren).
    cached_entries = [m for m in out if m["cached"]]
    sizes = await asyncio.gather(
        *(catalog.get_cached_size_bytes(m["model"], cfg.hf_home) for m in cached_entries)
    )
    for m, size in zip(cached_entries, sizes):
        m["size_bytes"] = size
    for m in out:
        m.setdefault("size_bytes", None)
    return sorted(out, key=lambda m: m["model"].lower())


def _model_history_with_current() -> list[dict]:
    """Verlauf wie 'ollama ps', aber historisch: laufende Session(s) zuerst
    (bei einem Hot Pool ggf. mehrere gleichzeitig), dann vergangene Sessions
    (Wechsel/Verdrängung/Entladen/Abstürze)."""
    history = list(process_manager.model_history)[:19]
    current = [
        {
            "model": eng.model,
            "loaded_at": eng.started_at,
            "unloaded_at": None,
            "duration_seconds": None,
            "reason": eng.state,  # "loading" oder "ready" markiert die laufende Session
            "error": None,
        }
        for eng in process_manager.engines.values()
    ]
    current.sort(key=lambda c: c["loaded_at"] or 0, reverse=True)
    return current + history


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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Manager – Live Status</title>
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
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; }
  .topbar-actions { display:flex; gap:8px; align-items:flex-start; flex:0 0 auto; }
  .sub { color: var(--text-dim); font-size: 13px; margin-bottom: 20px; }
  .conn { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--text-dim); }
  .dot { width:8px; height:8px; border-radius:50%; background: var(--bad); }
  .dot.live { background: var(--good); box-shadow: 0 0 6px var(--good); }
  #theme-toggle {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; width:36px; height:36px; font-size:16px; cursor:pointer; flex:0 0 auto;
  }
  #theme-toggle:hover { background:var(--panel-2); }
  #lang-select {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; height:36px; padding:0 8px; font-size:13px; cursor:pointer; flex:0 0 auto;
  }
  #lang-select:hover { background:var(--panel-2); }
  #rag-link, #config-link, #costs-link {
    display:inline-flex; align-items:center; background:var(--panel); border:1px solid var(--border);
    color:var(--text); text-decoration:none; border-radius:8px; height:36px; padding:0 12px;
    font-size:13px; flex:0 0 auto; box-sizing:border-box;
  }
  #rag-link:hover, #config-link:hover, #costs-link:hover { background:var(--panel-2); border-color: var(--accent); }
  .unload-btn {
    background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer;
  }
  .unload-btn:hover { border-color: var(--bad); color: var(--bad); }
  .unload-btn:disabled { opacity:.5; cursor:default; }
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
  .table-scroll { overflow-x:auto; }
  th, td { text-align:left; padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); }
  th { color:var(--text-dim); font-weight:600; font-size:11px; text-transform:uppercase; }
  tr:last-child td { border-bottom:none; }
  td.mono, th.mono { font-family: var(--mono); }
  .empty { color:var(--text-dim); font-size:13px; padding: 14px; text-align:center; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .bar-bg { background:var(--panel-2); border-radius:6px; height:6px; overflow:hidden; margin-top:6px; }
  .bar-fg { background:var(--accent); height:100%; transition: width .3s; }

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
  .modal .info-grid {
    display:grid; grid-template-columns: auto 1fr; gap:6px 14px;
    margin: 0 0 16px; padding:12px 14px; background:var(--panel-2);
    border-radius:8px; font-size:13px;
  }
  .modal .info-grid dt { color:var(--text-dim); margin:0; }
  .modal .info-grid dd { margin:0; font-family:var(--mono); }
  .modal .notes { font-size:12px; color:var(--text-dim); margin:0 0 16px; line-height:1.5; }
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
  .help-icon {
    display:inline-flex; align-items:center; justify-content:center;
    width:14px; height:14px; border-radius:50%; background:var(--panel-2);
    border:1px solid var(--border); color:var(--text-dim); font-size:10px;
    font-weight:700; cursor:pointer; margin-left:5px; vertical-align:middle;
    line-height:1; user-select:none; flex:0 0 auto;
  }
  .help-icon:hover { background:var(--accent-bg); color:var(--accent); border-color:var(--accent); }
  #help-modal-body { white-space:pre-line; line-height:1.6; font-size:14px; margin:0; }
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
      <h1 data-i18n="nav.title">vLLM Manager – Live Status</h1>
      <div class="sub"><span class="conn"><span class="dot" id="conn-dot"></span><span id="conn-text" data-i18n="nav.connecting">connecting…</span></span></div>
    </div>
    <div class="topbar-actions">
      <a href="/dashboard/config" id="config-link" data-i18n="nav.configLink">⚙️ Config →</a>
      <a href="/dashboard/costs" id="costs-link" data-i18n="nav.costsLink">💰 Costs →</a>
      <a href="/dashboard/rag" id="rag-link" data-i18n="nav.ragLink">RAG →</a>
      <select id="lang-select" data-i18n-title="lang.selectTitle" title="Language"></select>
      <button id="theme-toggle" data-i18n-title="theme.toggleTitle" title="Toggle theme">🌙</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label" data-i18n="card.loadedModels.label">Loaded Models</div>
      <div class="value" id="loaded-count">– / –</div>
      <div class="hint" id="loaded-list"></div>
    </div>
    <div class="card">
      <div class="label" data-i18n="card.lastPrompt.label">Last Prompt</div>
      <div class="value" id="last-prompt">–</div>
      <div class="hint" id="last-prompt-abs"></div>
    </div>
    <div class="card">
      <div class="label" data-i18n="card.requests.label">Requests running / waiting</div>
      <div class="value" id="running-waiting">–</div>
      <div class="hint" data-i18n="card.requests.hint">Sum across all loaded engines</div>
    </div>
    <div class="card">
      <div class="label"><span data-i18n="card.poolBudget.label">Pool Memory Budget (GPU)</span><span class="help-icon" data-help="poolBudget">?</span></div>
      <div class="value" id="pool-budget">–</div>
      <div class="bar-bg"><div class="bar-fg" id="pool-budget-bar" style="width:0%"></div></div>
    </div>
  </div>

  <section>
    <h2 data-i18n="section.systemUsage">System Usage</h2>
    <div class="chart-grid">
      <div class="card chart-card">
        <div class="label" data-i18n="section.gpuUsage">GPU Usage</div>
        <div class="value" id="gpu-percent">–</div>
        <div class="hint" id="gpu-extra"></div>
        <canvas id="gpu-chart" width="400" height="70"></canvas>
      </div>
      <div class="card chart-card">
        <div class="label"><span data-i18n="section.ramUsage">RAM Usage (Unified Memory)</span><span class="help-icon" data-help="ramUnified">?</span></div>
        <div class="value" id="ram-percent">–</div>
        <div class="hint" id="ram-extra"></div>
        <canvas id="ram-chart" width="400" height="70"></canvas>
      </div>
    </div>
  </section>

  <section>
    <h2 data-i18n="section.loadedModels">Loaded Models</h2>
    <div id="engines-box"></div>
  </section>

  <section>
    <h2 data-i18n="section.activeRequest">Active Request</h2>
    <div id="active-request-box"></div>
  </section>

  <section>
    <h2 data-i18n="section.recentRequests">Recent Requests</h2>
    <div id="recent-box"></div>
  </section>

  <section>
    <h2 data-i18n="section.modelHistory">Model History</h2>
    <table id="history-table" class="display" style="width:100%"></table>
  </section>

  <section>
    <h2 data-i18n="section.availableModels">Available Models</h2>
    <div class="model-grid" id="models-catalog-box"></div>
  </section>

  <section>
    <h2 data-i18n="section.downloads">Downloads (in progress)</h2>
    <div id="downloads-box"></div>
  </section>

  <div class="modal-overlay" id="model-modal-overlay">
    <div class="modal">
      <button class="close-btn" id="modal-close">✕</button>
      <h3 id="modal-title">–</h3>
      <div class="badges" id="modal-badges"></div>
      <div id="modal-actions"></div>
      <dl class="info-grid" id="modal-info"></dl>
      <p class="notes" id="modal-notes" style="display:none;"></p>
      <a class="hf-link" id="modal-hf-link" href="#" target="_blank" rel="noopener" data-i18n="modal.hfLink">🤗 View on HuggingFace ↗</a>
      <div class="json-label">
        <span data-i18n="modal.jsonLabel">VS Code Custom Endpoint (chatLanguageModels.json)</span>
        <button class="copy-btn" id="modal-copy-btn" data-i18n="modal.copy">Copy</button>
      </div>
      <pre id="modal-json"></pre>
    </div>
  </div>

  <div class="modal-overlay" id="help-modal-overlay">
    <div class="modal" style="max-width:480px;">
      <button class="close-btn" id="help-modal-close">✕</button>
      <h3 id="help-modal-title">–</h3>
      <p id="help-modal-body"></p>
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

// Live-Tabellen bekommen jede Sekunde per WebSocket frische Daten und
// wurden bisher IMMER komplett neu aufgebaut (innerHTML =), selbst wenn sich
// nichts geändert hat - das sah aus wie ein Sekunden-Reload und riss dabei
// jede laufende Textmarkierung in der Tabelle sofort wieder ab. safeSetHTML
// ersetzt das DOM nur, wenn sich der Inhalt tatsächlich geändert hat, und
// verschiebt ein Update sogar dann, wenn der Nutzer gerade Text innerhalb
// dieser Box markiert hat - der nächste Tick (≤1s später) holt es nach,
// sobald die Markierung beendet ist. Gibt zurück, ob das DOM ersetzt wurde
// (relevant für Aufrufer, die danach Event-Listener neu binden müssen).
function safeSetHTML(el, html) {
  if (!el) return false;
  if (el._lastHtml === html) return false;
  const sel = window.getSelection();
  if (sel && sel.rangeCount > 0 && !sel.isCollapsed && el.contains(sel.anchorNode)) {
    return false; // Nutzer markiert gerade Text hier drin - nicht unterbrechen.
  }
  el.innerHTML = html;
  el._lastHtml = html;
  return true;
}

// --- i18n ----------------------------------------------------------------
// Übersetzungen kommen vom Server (vllm_manager/languages/*.json), server-
// seitig hier als JS-Objekt eingebettet - kein Extra-Request nötig.
const TRANSLATIONS = __TRANSLATIONS_JSON__;
const DEFAULT_LANG = "en";
const LANG_NAMES = { en: "English", de: "Deutsch" };
let currentLang = localStorage.getItem("vllm_dashboard_lang");
if (!currentLang || !TRANSLATIONS[currentLang]) currentLang = DEFAULT_LANG;

function t(key, vars) {
  const table = TRANSLATIONS[currentLang] || {};
  const fallback = TRANSLATIONS[DEFAULT_LANG] || {};
  let s = table[key] ?? fallback[key] ?? key;
  if (vars) {
    for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
  }
  return s;
}
function localeFor(lang) { return lang === "de" ? "de-DE" : "en-US"; }

function populateLangSelect() {
  const sel = $("lang-select");
  sel.innerHTML = Object.keys(TRANSLATIONS).sort().map(code =>
    `<option value="${code}">${LANG_NAMES[code] || code}</option>`
  ).join("");
  sel.value = currentLang;
}
function applyStaticI18n() {
  document.documentElement.lang = currentLang;
  document.title = t("nav.title");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  updateConnText();
}
$("lang-select").addEventListener("change", (e) => {
  currentLang = e.target.value;
  localStorage.setItem("vllm_dashboard_lang", currentLang);
  applyStaticI18n();
  initHistoryTable();
  if (latestSnapshot) render(latestSnapshot);
});

function fmtAgo(seconds) {
  if (seconds === null || seconds === undefined) return t("ago.never");
  if (seconds < 2) return t("ago.justNow");
  if (seconds < 60) return t("ago.seconds", { n: Math.floor(seconds) });
  if (seconds < 3600) return t("ago.minutes", { n: Math.floor(seconds/60) });
  return t("ago.hours", { n: Math.floor(seconds/3600) });
}
function fmtMs(ms) { return (ms === null || ms === undefined) ? "–" : (ms < 1000 ? Math.round(ms) + " ms" : (ms/1000).toFixed(2) + " s"); }
function fmtPct(x) { return (x === null || x === undefined) ? "–" : Math.round(x*100) + "%"; }
// Belegter Speicherplatz eines Modells auf der Platte (catalog.get_cached_size_bytes).
function fmtGb(bytes) { return (bytes === null || bytes === undefined) ? "–" : (bytes/1e9).toFixed(1) + " GB"; }
// Fiktive Kosten (siehe cost_tracker.py) - kleine Beträge brauchen mehr
// Nachkommastellen, sonst rundet alles unter einem Cent auf $0.00.
function fmtUsd(x) {
  if (x === null || x === undefined) return null;
  if (x === 0) return "$0.00";
  return x < 0.01 ? "$" + x.toFixed(6) : "$" + x.toFixed(4);
}
// Human-readable duration, e.g. 3000s -> "50m", 65s -> "1m 5s", 45s -> "45s"
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
// Voller Modellname statt nur des letzten Pfad-Teils - zwei verschiedene
// Modelle können denselben Kurznamen haben (z.B. lokal quantisierte Modelle
// vs. HF-Repos mit demselben Dateinamen), das führt sonst zu Verwechslungen.
function modelName(m) { return m || "–"; }
// Welche App den Request geschickt hat, aus dem rohen User-Agent-Header (siehe
// telemetry.start_request) - unverändert angezeigt (keine Rate-/Vermutungslogik,
// um nichts falsch zu klassifizieren), lang truncated mit vollem Wert im Tooltip.
function appCell(userAgent) {
  if (!userAgent) return `<span class="hint">${esc(t("app.unknown"))}</span>`;
  const short = userAgent.length > 28 ? userAgent.slice(0, 27) + "…" : userAgent;
  return `<span title="${esc(userAgent)}">${esc(short)}</span>`;
}

// Zeigt, ob automatisches server-seitiges RAG für diese Anfrage gegriffen hat
// (siehe rag.apply_auto_rag / ModelConfig.rag_collection) - r.rag_used wird
// von telemetry.mark_rag_used() gesetzt, überlebt dank finish_request() auch
// den Übergang von Active in Recent Requests.
function ragCell(r) {
  if (!r.rag_used) return `<span class="hint">–</span>`;
  const title = t("rag.hitsTooltip", { collection: r.rag_collection, hits: r.rag_hits });
  return `<span class="badge ok" title="${esc(title)}">📚 ${esc(r.rag_collection)}</span>`;
}

// --- Hilfe-Icons (Fragezeichen neben Spaltenüberschriften/Labels) ----------
// Klickbares "?" öffnet ein Modal mit Erklärung, siehe help.<key>.title/body
// in den Übersetzungsdateien. Ein Icon kann von mehreren Stellen aus verlinkt
// werden (z.B. "loadTime" sowohl bei Aktiven als auch bei Letzten Anfragen).
function helpIcon(key) {
  return `<span class="help-icon" data-help="${key}" title="${esc(t("help.clickForInfo"))}">?</span>`;
}
function openHelpModal(key) {
  $("help-modal-title").textContent = t(`help.${key}.title`);
  $("help-modal-body").textContent = t(`help.${key}.body`);
  $("help-modal-overlay").classList.add("open");
}
// Event-Delegation statt Re-Binding bei jedem Render (die Tabellen werden
// jede Sekunde per WS-Heartbeat neu aufgebaut) - ein Listener reicht.
document.addEventListener("click", (e) => {
  const icon = e.target.closest(".help-icon");
  if (icon) { e.stopPropagation(); openHelpModal(icon.dataset.help); }
});
$("help-modal-close").addEventListener("click", () => $("help-modal-overlay").classList.remove("open"));
$("help-modal-overlay").addEventListener("click", (e) => {
  if (e.target.id === "help-modal-overlay") $("help-modal-overlay").classList.remove("open");
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("help-modal-overlay").classList.remove("open"); });
function reasonLabel(r) {
  if (r === "ready") return t("reason.ready");
  if (r === "loading") return t("reason.loading");
  if (r === "manual_unload") return t("reason.manual_unload");
  if (r === "idle_timeout") return t("reason.idle_timeout");
  if (r === "crashed") return t("reason.crashed");
  if (r === "timeout") return t("reason.timeout");
  if (r === "shutdown") return t("reason.shutdown");
  if (r === "restart") return t("reason.restart");
  if (r === "failed_to_start") return t("reason.failedToStart");
  if (r && r.startsWith("replaced_by:")) return t("reason.replacedBy", { model: esc(modelName(r.slice(13))) });
  if (r && r.startsWith("evicted_for:")) return t("reason.evictedFor", { model: esc(modelName(r.slice(12))) });
  return esc(r || "–");
}
function reasonBadgeClass(r) {
  if (r === "ready") return "ok";
  if (r === "loading") return "loading";
  if (r === "crashed" || r === "timeout" || r === "failed_to_start") return "error";
  if (r && r.startsWith("evicted_for:")) return "running";
  return "idle";
}
function jobStateLabel(s) { return (TRANSLATIONS[currentLang] || {})["job." + s] !== undefined || (TRANSLATIONS[DEFAULT_LANG] || {})["job." + s] !== undefined ? t("job." + s) : esc(s); }

// --- Model History (DataTables, siehe /static/vendor/datatables/README.md) -
// Rollierendes Log (process_manager.MAX_HISTORY=50) - wächst laufend weiter,
// daher paginiert statt als eine lange Liste. Spaltentitel/Leertext hängen
// von der Sprache ab -> bei Sprachwechsel komplett neu aufgebaut (siehe
// lang-select-Handler unten), statt Header live zu patchen.
let historyTable = null;
let historyFingerprint = null;

function initHistoryTable() {
  if (historyTable) { historyTable.destroy(); $("history-table").innerHTML = ""; }
  historyFingerprint = null;
  historyTable = new DataTable("#history-table", {
    data: [],
    order: [],
    pageLength: 10,
    layout: { bottomEnd: "inputPaging" },
    language: { emptyTable: t("empty.noHistory") },
    columns: [
      { title: t("th.model"), data: null, render: (h, type) => type === "display" ? esc(modelName(h.model)) : (modelName(h.model) || "") },
      {
        title: t("th.status"), data: null, render: (h, type) => {
          if (type !== "display") return h.reason || "";
          const errHint = h.error ? `<div class="hint" style="margin-top:4px;">${esc(h.error.split("\n")[0])}</div>` : "";
          return `<span class="badge ${reasonBadgeClass(h.reason)}">${reasonLabel(h.reason)}</span>${errHint}`;
        },
      },
      {
        title: t("th.loadedAt"), data: null, render: (h, type) => {
          if (type !== "display") return h.loaded_at ?? 0;
          return h.loaded_at ? new Date(h.loaded_at * 1000).toLocaleTimeString(localeFor(currentLang)) : "–";
        },
      },
      {
        title: t("th.unloadedAt"), data: null, render: (h, type) => {
          const ongoing = h.unloaded_at === null || h.unloaded_at === undefined;
          if (type !== "display") return ongoing ? Number.MAX_SAFE_INTEGER : h.unloaded_at;
          return ongoing ? "–" : new Date(h.unloaded_at * 1000).toLocaleTimeString(localeFor(currentLang));
        },
      },
      {
        title: t("th.duration"), data: null, render: (h, type) => {
          const ongoing = h.unloaded_at === null || h.unloaded_at === undefined;
          const seconds = ongoing ? (h.loaded_at ? (Date.now() / 1000 - h.loaded_at) : null) : h.duration_seconds;
          return type !== "display" ? (seconds ?? 0) : fmtDuration(seconds);
        },
      },
    ],
  });
}

function updateHistoryTable(hist) {
  const fp = JSON.stringify(hist);
  if (fp === historyFingerprint) return;
  historyFingerprint = fp;
  historyTable.clear();
  historyTable.rows.add(hist);
  historyTable.draw(false);
}

// Status einer abgeschlossenen Anfrage (Letzte Anfragen) - "aborted_loop"
// (siehe main.py _has_repetition_loop) bekommt eine eigene, erkennbare
// Markierung statt generisch als "Fehler" durchzugehen.
function statusCell(r) {
  if (r.status === "ok") return `<span class="badge ok">${t("status.ok")}</span>`;
  if (r.status === "aborted_loop") return `<span class="badge error" title="${esc(t("status.abortedLoopHint"))}">${t("status.abortedLoop")}</span>`;
  return `<span class="badge error">${t("status.error")}</span>`;
}

// Phasen einer aktiven Anfrage (siehe telemetry.py _set_phase): was die
// Engine gerade tut, für die Active-Requests-Tabelle im Dashboard.
const PHASE_META = {
  loading:   { icon: "🥶", key: "phase.loading",   badgeClass: "loading" },
  prefill:   { icon: "⏳", key: "phase.prefill",   badgeClass: "idle" },
  thinking:  { icon: "💭", key: "phase.thinking",  badgeClass: "running" },
  tool_call: { icon: "🔧", key: "phase.toolCall",  badgeClass: "running" },
  generating:{ icon: "✍️", key: "phase.generating",badgeClass: "running" },
};
function phaseInfo(phase) {
  const meta = PHASE_META[phase] || { icon: "•", key: null, badgeClass: "idle" };
  return { icon: meta.icon, badgeClass: meta.badgeClass, label: () => meta.key ? t(meta.key) : (phase || "–") };
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
  latestSnapshot = data;
  const engs = data.engines || [];
  const pool = data.pool || {};

  $("loaded-count").textContent = engs.length + " / " + (pool.slots_total ?? engs.length);
  $("loaded-list").textContent = engs.length
    ? engs.map(e => modelName(e.loaded_model)).join(", ")
    : t("card.loadedModels.none");

  if (engs.length === 0) {
    safeSetHTML($("engines-box"), `<div class="empty">${t("empty.noModelLoaded")}</div>`);
  } else {
    safeSetHTML($("engines-box"), `<div class="table-scroll"><table><thead><tr>
      <th>${t("th.model")}</th><th>${t("th.status")}</th><th>${t("th.port")}</th><th>${t("th.since")}</th><th>${t("th.requests")}${helpIcon("requestsEngine")}</th><th>${t("th.kvCache")}${helpIcon("kvCache")}</th><th>${t("th.avgTtft")}${helpIcon("avgTtft")}</th><th>${t("th.avgMsPerTok")}${helpIcon("avgMsPerTok")}</th><th>${t("th.tokensPromptGen")}${helpIcon("tokensPromptGen")}</th><th>${t("th.action")}</th>
      </tr></thead><tbody>` + engs.map(e => {
        const m = e.metrics || {};
        let badge;
        if (e.state === "loading") badge = `<span class="badge loading">${t("badge.coldStart")}</span>`;
        else if (e.state === "ready") badge = `<span class="badge ok">${t("badge.ready")}</span>`;
        else badge = `<span class="badge idle">${t("badge.idle")}</span>`;
        const since = e.running
          ? t(e.state === "loading" ? "engine.loadingSince" : "engine.runningSince", { duration: fmtDuration(e.uptime_seconds) })
          : (e.last_error ? t("engine.error", { msg: esc(e.last_error.split("\n")[0]) }) : "–");
        return `<tr>
          <td>${esc(modelName(e.loaded_model))}</td>
          <td>${badge}</td>
          <td class="mono">${e.port ?? "–"}</td>
          <td class="mono">${since}</td>
          <td class="mono">${(m.num_requests_running ?? "–") + " / " + (m.num_requests_waiting ?? "–")}</td>
          <td class="mono">${fmtPct(m.kv_cache_usage_perc)}</td>
          <td class="mono">${fmtMs(m.avg_ttft_ms)}</td>
          <td class="mono">${fmtMs(m.avg_tpot_ms)}</td>
          <td class="mono">${(m.prompt_tokens_total ?? "–") + " / " + (m.generation_tokens_total ?? "–")}</td>
          <td><button class="unload-btn" data-model="${esc(e.loaded_model)}">${t("action.unload")}</button></td>
        </tr>`;
      }).join("") + `</tbody></table></div>`);
  }

  $("last-prompt").textContent = fmtAgo(data.seconds_since_last_request);
  $("last-prompt-abs").textContent = data.last_request_at
    ? new Date(data.last_request_at*1000).toLocaleString(localeFor(currentLang)) : "";

  const totRunning = engs.reduce((s, e) => s + ((e.metrics || {}).num_requests_running || 0), 0);
  const totWaiting = engs.reduce((s, e) => s + ((e.metrics || {}).num_requests_waiting || 0), 0);
  $("running-waiting").textContent = engs.length ? (totRunning + " / " + totWaiting) : "– / –";

  const usedPct = pool.ceiling ? Math.round((pool.used || 0) * 100) : null;
  const ceilPct = pool.ceiling ? Math.round(pool.ceiling * 100) : null;
  $("pool-budget").textContent = usedPct !== null ? t("poolBudget.value", { used: usedPct, ceiling: ceilPct }) : "–";
  $("pool-budget-bar").style.width = (pool.ceiling ? Math.min(100, Math.round((pool.used / pool.ceiling) * 100)) : 0) + "%";

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
    safeSetHTML($("active-request-box"), `<div class="empty">${t("empty.noActiveRequest")}</div>`);
  } else {
    safeSetHTML($("active-request-box"), `<div class="table-scroll"><table><thead><tr>
      <th>${t("th.model")}</th><th>${t("th.app")}</th><th>${t("th.endpoint")}</th><th>${t("th.port")}</th><th>${t("th.rag")}${helpIcon("rag")}</th><th>${t("th.phase")}${helpIcon("phase")}</th><th>${t("th.elapsed")}</th><th>${t("th.loadTime")}${helpIcon("loadTime")}</th><th>${t("th.ttft")}${helpIcon("ttft")}</th><th>${t("th.tokensPromptGen")}${helpIcon("liveTokens")}</th><th>${t("th.reasoningTokens")}${helpIcon("liveTokens")}</th><th>${t("th.throughput")}${helpIcon("throughput")}</th><th>${t("th.cost")}${helpIcon("cost")}</th>
      </tr></thead><tbody>` + active.map(r => {
        const elapsed = Date.now()/1000 - r.started_at;
        const port = (engs.find(e => e.loaded_model === r.model) || {}).port;
        // queued_ms wird erst gesetzt, sobald das Modell bereit ist (siehe
        // telemetry.mark_ready) - bis dahin wartet die Anfrage auf einen
        // Kaltstart/Modellwechsel, es wird also noch nichts generiert.
        const loading = r.queued_ms === null || r.queued_ms === undefined;
        const genElapsedSec = loading ? null : Math.max(0, elapsed - r.queued_ms / 1000);
        const throughput = (!loading && r.tokens_streamed > 0 && genElapsedSec > 0.05)
          ? (r.tokens_streamed / genElapsedSec).toFixed(1) + " tok/s"
          : "–";
        const history = r.phase_history || [];
        const lastChange = history.length ? history[history.length - 1] : null;
        const phaseSinceSec = lastChange ? Math.max(0, Date.now()/1000 - lastChange.at) : elapsed;
        const pi = phaseInfo(r.phase);
        // Hover-Tooltip: kleine Zeitleiste aller Phasenwechsel seit Requeststart.
        const timeline = history.map((h, i) => {
          const from = fmtDuration(h.at - r.started_at);
          const to = (i + 1 < history.length) ? fmtDuration(history[i + 1].at - r.started_at) : t("phase.now");
          return `${phaseInfo(h.phase).label()} ${from} → ${to}`;
        }).join("\n");
        return `<tr>
          <td>${esc(modelName(r.model))}</td>
          <td class="mono">${appCell(r.user_agent)}</td>
          <td class="mono">${esc(r.path || "–")}${r.is_stream ? ` <span class="badge idle">${t("badge.stream")}</span>` : ""}</td>
          <td class="mono">${port ?? "–"}</td>
          <td>${ragCell(r)}</td>
          <td>
            <span class="badge ${pi.badgeClass}" title="${esc(timeline)}">${pi.icon} ${esc(pi.label())}</span>
            <div class="hint">${esc(t("phase.since", { duration: fmtDuration(phaseSinceSec) }))}</div>
          </td>
          <td class="mono">${fmtDuration(elapsed)}</td>
          <td class="mono">${r.queued_ms ? fmtMs(r.queued_ms) : "–"}</td>
          <td class="mono">${fmtMs(r.ttft_ms)}</td>
          <td class="mono">${r.prompt_tokens ?? "–"} / ${r.tokens_streamed ?? 0}</td>
          <td class="mono">${r.reasoning_tokens_streamed ?? 0}</td>
          <td class="mono">${throughput}</td>
          <td class="mono" title="${esc(t("cost.hint.soFarOutputOnly"))}">${fmtUsd(r.estimated_output_cost_usd) ?? "–"} <span class="hint">${t("cost.soFar")}</span></td>
        </tr>`;
      }).join("") + `</tbody></table></div>`);
  }

  updateHistoryTable(data.model_history || []);

  const dls = data.downloads || [];
  if (dls.length === 0) {
    safeSetHTML($("downloads-box"), `<div class="empty">${t("empty.noDownloads")}</div>`);
  } else {
    safeSetHTML($("downloads-box"), `<div class="table-scroll"><table><thead><tr>
      <th>${t("th.model")}</th><th>${t("th.status")}</th><th>${t("th.progress")}</th><th>${t("th.speed")}</th><th>${t("th.eta")}</th><th>${t("th.action")}</th>
      </tr></thead><tbody>` + dls.map(j => {
        const cancellable = j.state === "queued" || j.state === "resolving" || j.state === "downloading";
        return `<tr>
        <td>${esc(j.model)}</td>
        <td>${jobStateLabel(j.state)}</td>
        <td class="mono">${Math.min(j.percent,100)}% (${(Math.min(j.bytes_done,j.bytes_total)/1e9).toFixed(1)}/${(j.bytes_total/1e9).toFixed(1)} GB)</td>
        <td class="mono">${j.speed_mbps} MB/s</td>
        <td class="mono">${fmtDuration(j.eta_seconds)}</td>
        <td>${cancellable ? `<button class="btn danger cancel-download-btn" data-job-id="${esc(j.job_id)}" data-model="${esc(j.model)}">${t("action.cancelDownload")}</button>` : "–"}</td>
      </tr>`;
      }).join("") + `</tbody></table></div>`);
  }

  const recent = data.recent_requests || [];
  if (recent.length === 0) {
    safeSetHTML($("recent-box"), `<div class="empty">${t("empty.noRecentRequests")}</div>`);
  } else {
    safeSetHTML($("recent-box"), `<div class="table-scroll"><table><thead><tr>
      <th>${t("th.time")}</th><th>${t("th.model")}</th><th>${t("th.app")}</th><th>${t("th.rag")}${helpIcon("rag")}</th><th>${t("th.status")}</th><th>${t("th.loadTime")}${helpIcon("loadTime")}</th><th>${t("th.duration")}</th><th>${t("th.ttft")}${helpIcon("ttft")}</th><th>${t("th.promptTokens")}${helpIcon("requestTokens")}</th><th>${t("th.complTokens")}${helpIcon("requestTokens")}</th><th>${t("th.throughput")}${helpIcon("avgThroughput")}</th><th>${t("th.cost")}${helpIcon("cost")}</th>
      </tr></thead><tbody>` + recent.map(r => {
        // Gleiche Formel wie bei Active Requests (siehe "throughput" oben):
        // Tokens pro Sekunde SEIT Modell bereit war (duration_ms abzüglich
        // queued_ms = Warten auf Kaltstart/Modellwechsel) - nicht ab
        // Request-Beginn, sonst würde ein langsam ladendes Modell die
        // Generierungsgeschwindigkeit künstlich schlecht aussehen lassen.
        const genElapsedSec = r.duration_ms != null ? Math.max(0, (r.duration_ms - (r.queued_ms || 0)) / 1000) : null;
        const avgThroughput = (r.completion_tokens > 0 && genElapsedSec > 0.05)
          ? (r.completion_tokens / genElapsedSec).toFixed(1) + " tok/s"
          : "–";
        return `<tr>
        <td class="mono">${new Date(r.started_at*1000).toLocaleTimeString(localeFor(currentLang))}</td>
        <td>${esc(modelName(r.model))}</td>
        <td class="mono">${appCell(r.user_agent)}</td>
        <td>${ragCell(r)}</td>
        <td>${statusCell(r)}</td>
        <td class="mono">${r.queued_ms ? fmtMs(r.queued_ms) : "–"}</td>
        <td class="mono">${fmtMs(r.duration_ms)}</td>
        <td class="mono">${fmtMs(r.ttft_ms)}</td>
        <td class="mono">${r.prompt_tokens ?? "–"}</td>
        <td class="mono">${r.completion_tokens ?? "–"}</td>
        <td class="mono">${avgThroughput}</td>
        <td class="mono">${fmtUsd(r.cost_usd) ?? "–"}</td>
      </tr>`;
      }).join("") + `</tbody></table></div>`);
  }

  latestCatalog = data.models_catalog || [];
  const catalog = latestCatalog;
  if (catalog.length === 0) {
    safeSetHTML($("models-catalog-box"), `<div class="empty">${t("empty.noModelsKnown")}</div>`);
  } else {
    const catalogHtml = catalog.map((m, i) => `
      <div class="model-item" data-idx="${i}">
        <div class="name">${esc(modelName(m.model))}</div>
        <div class="badges">
          ${m.loaded ? `<span class="badge running">${t("badge.loaded")}</span>` : ""}
          <span class="badge ${m.cached ? 'ok' : 'idle'}">${m.cached ? t("badge.cached") : t("badge.notCached")}</span>
          ${!m.enabled ? `<span class="badge error">${t("badge.disabled")}</span>` : ""}
          ${m.task === "embed" ? `<span class="badge idle">${t("badge.embed")}</span>` : ""}
          ${m.tool_calling ? `<span class="badge idle">${t("badge.toolCalling")}</span>` : ""}
          ${m.reasoning ? `<span class="badge idle">${t("badge.reasoning")}</span>` : ""}
          ${m.vision ? `<span class="badge idle">${t("badge.vision")}</span>` : ""}
        </div>
        ${m.size_bytes != null ? `<div class="hint">💾 ${fmtGb(m.size_bytes)}</div>` : ""}
      </div>`).join("");
    // Event-Listener nur neu binden, wenn safeSetHTML das DOM auch wirklich
    // ersetzt hat (sonst hängen die alten Listener noch am unveränderten DOM).
    if (safeSetHTML($("models-catalog-box"), catalogHtml)) {
      document.querySelectorAll("#models-catalog-box .model-item").forEach(el => {
        el.addEventListener("click", () => openModal(latestCatalog[parseInt(el.dataset.idx, 10)]));
      });
    }
  }
}

// --- Modell manuell laden (Dashboard-Button, Gegenstück zu unloadModel) --
// Ruft .../load?background=true auf - der Kaltstart läuft serverseitig als
// überwachter Hintergrund-Task (siehe main.py _background_load), der Fetch
// hier kehrt also sofort zurück statt bei großen Modellen minutenlang auf
// die HTTP-Antwort zu warten. Fortschritt zeigt wie gewohnt der nächste
// WebSocket-Heartbeat (Geladene Modelle/Modell-Verlauf: "lädt gerade").
async function loadModel(model, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t("action.loading");
  try {
    const headers = {};
    if (apiKey) headers["Authorization"] = "Bearer " + apiKey;
    const res = await fetch(`/models/${encodeURIComponent(model)}/load?background=true`, { method: "POST", headers });
    if (!res.ok) throw new Error(await res.text());
    closeModal();
    // Nächster Heartbeat (≤1s) zeigt den Kaltstart in der Tabelle an.
  } catch (e) {
    alert(t("error.loadFailed", { msg: e.message }));
    btn.disabled = false;
    btn.textContent = original;
  }
}

// --- Modell manuell entladen (Dashboard-Button) --------------------------
async function unloadModel(model, btn) {
  if (!confirm(t("confirm.unload", { model: modelName(model) }))) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t("action.unloading");
  try {
    const headers = {};
    if (apiKey) headers["Authorization"] = "Bearer " + apiKey;
    const res = await fetch(`/models/${encodeURIComponent(model)}/unload`, { method: "POST", headers });
    if (!res.ok) throw new Error(await res.text());
    // Nächster Heartbeat (≤1s) aktualisiert die Tabelle automatisch.
  } catch (e) {
    alert(t("error.unloadFailed", { msg: e.message }));
    btn.disabled = false;
    btn.textContent = original;
  }
}
// --- Lokale Modell-Dateien unwiderruflich von der Platte löschen ---------
// Trifft sowohl registrierte (config.json) als auch nur lokal gecachte, gar
// nicht (mehr) registrierte Modelle - siehe main.py DELETE /models/{model}/cache.
// Bewusst zwei Schritte: erst Größe abfragen (damit man weiß, was man da
// löscht), dann erst der eigentliche, unwiderrufliche Löschvorgang.
async function deleteModelCache(model, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t("action.checkingSize");
  const headers = {};
  if (apiKey) headers["Authorization"] = "Bearer " + apiKey;
  try {
    const infoRes = await fetch(`/models/${encodeURIComponent(model)}/cache_info`, { headers });
    if (!infoRes.ok) throw new Error(await infoRes.text());
    const info = await infoRes.json();
    if (!info.cached) {
      alert(t("info.nothingCachedToDelete", { model: modelName(model) }));
      btn.disabled = false;
      btn.textContent = original;
      return;
    }
    const gb = (info.size_bytes / 1e9).toFixed(1);
    if (!confirm(t("confirm.deleteModelCache", { model: modelName(model), size: gb }))) {
      btn.disabled = false;
      btn.textContent = original;
      return;
    }
    btn.textContent = t("action.deleting");
    const delRes = await fetch(`/models/${encodeURIComponent(model)}/cache`, { method: "DELETE", headers });
    if (!delRes.ok) throw new Error(await delRes.text());
    closeModal();
    // Nächster Heartbeat (≤1s) entfernt das Modell aus dem Katalog automatisch.
  } catch (e) {
    alert(t("error.deleteCacheFailed", { msg: e.message }));
    btn.disabled = false;
    btn.textContent = original;
  }
}

$("engines-box").addEventListener("click", (e) => {
  const btn = e.target.closest(".unload-btn");
  if (btn) unloadModel(btn.dataset.model, btn);
});

// --- Laufenden Download abbrechen ----------------------------------------
// Event-Delegation auf dem stabilen Container statt direkter Listener pro
// Button - downloads-box wird bei jedem Tick per safeSetHTML() ggf. neu
// aufgebaut, der Container selbst aber nie ersetzt.
async function cancelDownload(jobId, model, btn) {
  if (!confirm(t("confirm.cancelDownload", { model: modelName(model) }))) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t("action.cancelling");
  try {
    const headers = {};
    if (apiKey) headers["Authorization"] = "Bearer " + apiKey;
    const res = await fetch(`/models/pull/${encodeURIComponent(jobId)}/cancel`, { method: "POST", headers });
    if (!res.ok) throw new Error(await res.text());
    // Nächster Heartbeat (≤1s) aktualisiert die Tabelle automatisch.
  } catch (e) {
    alert(t("error.cancelDownloadFailed", { msg: e.message }));
    btn.disabled = false;
    btn.textContent = original;
  }
}
$("downloads-box").addEventListener("click", (e) => {
  const btn = e.target.closest(".cancel-download-btn");
  if (btn) cancelDownload(btn.dataset.jobId, btn.dataset.model, btn);
});

// --- Modell-Katalog / Klick-Modal ---------------------------------------
let latestCatalog = [];

function openModal(m) {
  $("modal-title").textContent = m.model;
  $("modal-hf-link").href = "https://huggingface.co/" + m.model;
  $("modal-badges").innerHTML = [
    m.loaded ? `<span class="badge running">${t("badge.loaded")}</span>` : "",
    `<span class="badge ${m.cached ? 'ok' : 'idle'}">${m.cached ? t("badge.cached") : t("badge.notCached")}</span>`,
    !m.enabled ? `<span class="badge error">${t("badge.disabled")}</span>` : `<span class="badge ok">${t("badge.enabled")}</span>`,
  ].filter(Boolean).join("");

  const yesNo = (v) => v ? t("common.yes") : t("common.no");
  const infoRows = [
    [t("modal.contextLength"), m.max_model_len != null ? m.max_model_len.toLocaleString(localeFor(currentLang)) + " " + t("modal.tokens") : "–"],
    [t("cfg.field.task"), m.task === "embed" ? t("badge.embed") : m.task || "–"],
    [t("badge.toolCalling"), yesNo(m.tool_calling)],
    [t("badge.reasoning"), yesNo(m.reasoning)],
    [t("badge.vision"), yesNo(m.vision)],
    [t("modal.gpuMemory"), m.gpu_memory_utilization != null ? fmtPct(m.gpu_memory_utilization) : "–"],
    [t("modal.diskSize"), fmtGb(m.size_bytes)],
  ];
  $("modal-info").innerHTML = infoRows.map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join("");

  if (m.notes) {
    $("modal-notes").style.display = "block";
    $("modal-notes").textContent = m.notes;
  } else {
    $("modal-notes").style.display = "none";
  }

  $("modal-actions").innerHTML = (m.loaded || m.cached || m.enabled) ? `<div style="margin-bottom:16px; display:flex; gap:8px; flex-wrap:wrap;">
    ${(!m.loaded && m.enabled) ? `<button class="btn" id="modal-load-btn">${t("action.load")}</button>` : ""}
    ${m.loaded ? `<button class="unload-btn" id="modal-unload-btn">${t("action.unload")}</button>` : ""}
    ${m.cached ? `<button class="btn danger" id="modal-delete-cache-btn" ${m.loaded ? "disabled title=\"" + esc(t("hint.unloadBeforeDelete")) + "\"" : ""}>${t("action.deleteFromDisk")}</button>` : ""}
  </div>` : "";
  if (!m.loaded && m.enabled) {
    $("modal-load-btn").addEventListener("click", () => loadModel(m.model, $("modal-load-btn")));
  }
  if (m.loaded) {
    $("modal-unload-btn").addEventListener("click", () => unloadModel(m.model, $("modal-unload-btn")));
  }
  if (m.cached && !m.loaded) {
    $("modal-delete-cache-btn").addEventListener("click", () => deleteModelCache(m.model, $("modal-delete-cache-btn")));
  }

  const url = `${location.protocol}//${location.host}/v1`;
  // maxInputTokens + maxOutputTokens dürfen NICHT beide = max_model_len sein:
  // vLLM lehnt input_tokens + output_tokens > max_model_len ab (400 Bad
  // Request), und VS Code füllt den Input praktisch bis maxInputTokens auf -
  // ohne Reserve für die Ausgabe schlägt dann jeder etwas längere Prompt fehl.
  // Deshalb ein festes Ausgabe-Budget von der Kontextlänge abziehen.
  const outputBudget = Math.max(512, Math.min(4096, Math.floor(m.max_model_len / 4)));
  const entry = {
    id: m.model,
    name: modelName(m.model),
    url: url,
    toolCalling: !!m.tool_calling,
    vision: !!m.vision,
    maxInputTokens: Math.max(1, m.max_model_len - outputBudget),
    maxOutputTokens: outputBudget,
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
// navigator.clipboard gibt es nur in "sicheren Kontexten" (HTTPS oder
// localhost) - dieses Dashboard läuft absichtlich über eine reine
// HTTP-LAN-IP (siehe Sicherheit in Anleitung.md), da ist die Clipboard-API
// im Browser gar nicht vorhanden. Fallback über eine unsichtbare Textarea +
// execCommand("copy"), die auch über HTTP funktioniert.
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
      ok ? resolve() : reject(new Error("execCommand('copy') fehlgeschlagen"));
    } catch (e) {
      document.body.removeChild(ta);
      reject(e);
    }
  });
}
$("modal-copy-btn").addEventListener("click", () => {
  copyText($("modal-json").textContent).then(() => {
    const btn = $("modal-copy-btn");
    const old = btn.textContent;
    btn.textContent = t("modal.copied");
    setTimeout(() => { btn.textContent = old; }, 1500);
  }).catch((e) => {
    alert(t("error.generic", { msg: e.message }));
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
let latestSnapshot = null;
let connState = "connecting"; // "connecting" | "live" | "reconnecting"

function updateConnText() { $("conn-text").textContent = t("nav." + connState); }

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/dashboard/ws");

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
    connState = "reconnecting";
    updateConnText();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

populateLangSelect();
applyStaticI18n();
initHistoryTable();
connect();
</script>
</body>
</html>
"""

DASHBOARD_HTML = DASHBOARD_HTML.replace("__TRANSLATIONS_JSON__", _LANGUAGES_JS)
