"""FastAPI-App: OpenAI-kompatibler Proxy mit Auto-Load, Modell-Verwaltung,
Download-Endpoints mit Fortschritt, und gemounteter MCP-Server unter /mcp."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from pydantic import ValidationError

from . import config_editor, downloader, process_manager, rag, telemetry
from .auth import ApiKeyMiddleware
from .catalog import list_cached_models
from .config import get_config
from .config_dashboard import router as config_dashboard_router
from .dashboard import router as dashboard_router
from .mcp_tools import mcp
from .ollama_compat import router as ollama_router
from .rag_dashboard import router as rag_dashboard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vllm_manager")

IDLE_CHECK_INTERVAL = 30


async def _idle_watchdog() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL)
        cfg = get_config()
        if not cfg.idle_timeout_seconds:
            continue
        now = time.time()
        for eng in list(process_manager.engines.values()):
            if now - eng.last_used > cfg.idle_timeout_seconds:
                logger.info("Idle-Timeout (%ss) erreicht, entlade %s", cfg.idle_timeout_seconds, eng.model)
                await process_manager.stop_engine(eng.model, reason="idle_timeout")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_editor.load_config_with_fallback()
    watchdog = asyncio.create_task(_idle_watchdog())
    async with mcp.session_manager.run():
        yield
    watchdog.cancel()
    await process_manager.stop_engine(reason="shutdown")


app = FastAPI(title="vLLM Manager", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())
app.include_router(dashboard_router)
app.include_router(ollama_router)
app.include_router(rag_dashboard_router)
app.include_router(config_dashboard_router)
# Reine ASGI-Middleware statt @app.middleware("http") - siehe auth.py
# Docstring: BaseHTTPMiddleware bricht Streaming-Responses (stream: true).
app.add_middleware(ApiKeyMiddleware)


@app.get("/health")
async def health():
    return {"status": "ok", "engines": [e.status() for e in process_manager.engines.values()]}


@app.get("/models")
async def list_models_endpoint():
    cfg = get_config()
    cached = set(list_cached_models(cfg.hf_home))
    out = []
    for name, mcfg in cfg.models.items():
        out.append({
            "model": name,
            "cached": name in cached,
            "loaded": process_manager.is_ready(name),
            "enabled": mcfg.enabled,
            "notes": mcfg.notes,
        })
    known = {m["model"] for m in out}
    for name in sorted(cached - known):
        out.append({
            "model": name,
            "cached": True,
            "loaded": process_manager.is_ready(name),
            "enabled": True,
            "notes": "Lokal gecacht, aber nicht in config.json registriert.",
        })
    return {"models": out, "default_model": cfg.default_model, "active": process_manager.loaded_models()}


@app.post("/models/pull")
async def pull_model_endpoint(body: dict):
    model = body.get("model")
    if not model:
        raise HTTPException(400, "'model' fehlt im Request-Body, z.B. {\"model\": \"org/name\"}.")
    job_id = await downloader.start_job(model, body.get("revision"), body.get("hf_token"))
    return {"job_id": job_id}


@app.get("/models/pull")
async def list_pull_jobs():
    return {"jobs": downloader.list_jobs()}


@app.get("/models/pull/{job_id}")
async def pull_status_endpoint(job_id: str):
    job = downloader.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unbekannte job_id.")
    return job


@app.post("/models/{model:path}/load")
async def load_model_endpoint(model: str):
    try:
        return await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        raise HTTPException(500, str(e))


@app.post("/models/{model:path}/unload")
async def unload_model_endpoint(model: str):
    if model in process_manager.engines:
        await process_manager.stop_engine(model)
        return {"status": "unloaded"}
    return {"status": "not_loaded", "currently_loaded": process_manager.loaded_models()}


# --- RAG (Retrieval-Augmented Generation) ---------------------------------
# Backend für die Dashboard-Seite /dashboard/rag. Dieselbe Logik ist auch über
# die rag_*-MCP-Tools nutzbar (siehe mcp_tools.py) - beide rufen rag.py auf.


@app.get("/rag/collections")
async def rag_list_collections_endpoint():
    try:
        return {"collections": await rag.list_collections()}
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.get("/rag/collections/{collection}/documents")
async def rag_list_documents_endpoint(collection: str):
    try:
        return {"collection": collection, "documents": await rag.list_documents(collection)}
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.post("/rag/collections/{collection}/text")
async def rag_add_text_endpoint(collection: str, body: dict):
    text = body.get("text")
    if not text:
        raise HTTPException(400, "'text' fehlt im Request-Body.")
    try:
        return await rag.add_text(collection, text, source=body.get("source", "text"))
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.post("/rag/collections/{collection}/file")
async def rag_add_file_endpoint(collection: str, body: dict):
    path = body.get("path")
    if not path:
        raise HTTPException(400, "'path' fehlt im Request-Body.")
    try:
        return await rag.add_file(collection, path)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/rag/collections/{collection}/search")
async def rag_search_endpoint(collection: str, body: dict):
    query = body.get("query")
    if not query:
        raise HTTPException(400, "'query' fehlt im Request-Body.")
    try:
        results = await rag.search(collection, query, int(body.get("top_k", 5)))
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))
    return {"collection": collection, "results": results}


@app.delete("/rag/collections/{collection}/documents/{document_id}")
async def rag_delete_document_endpoint(collection: str, document_id: str):
    try:
        return await rag.delete_document(collection, document_id)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


@app.delete("/rag/collections/{collection}")
async def rag_delete_collection_endpoint(collection: str):
    try:
        return await rag.delete_collection(collection)
    except rag.RagNotConfigured as e:
        raise HTTPException(400, str(e))


# --- Config-Editor (/dashboard/config) -------------------------------------
# Backend für die Config-Editor-Seite: liest/schreibt config.json über
# config_editor.py (Validierung + automatisches Backup + Live-Übernahme ohne
# Neustart für alles außer host/port, siehe dortige Docstrings).

# Felder, die nur beim Neustart des Manager-Prozesses selbst greifen (der
# Bind-Socket von uvicorn) - alles andere liest jeder Request/jeder neue
# Engine-Start ohnehin frisch über get_config(), ein Restart ist dafür nicht
# nötig.
RESTART_REQUIRED_FIELDS = {"host", "port"}


@app.get("/config")
async def get_config_endpoint():
    return {
        "config": get_config().model_dump(),
        "startup_warning": config_editor.startup_warning,
    }


@app.post("/config")
async def save_config_endpoint(body: dict):
    old_dump = get_config().model_dump()
    try:
        new_cfg, backup_name = config_editor.save_config(body)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    new_dump = new_cfg.model_dump()
    restart_recommended = any(old_dump.get(f) != new_dump.get(f) for f in RESTART_REQUIRED_FIELDS)
    return {"ok": True, "backup": backup_name, "restart_recommended": restart_recommended}


@app.get("/config/backups")
async def list_config_backups_endpoint():
    return {"backups": config_editor.list_backups()}


@app.post("/config/restore")
async def restore_config_backup_endpoint(body: dict):
    filename = body.get("filename")
    if not filename:
        raise HTTPException(400, "'filename' fehlt im Request-Body.")
    try:
        new_cfg, backup_name = config_editor.restore_backup(filename)
    except FileNotFoundError:
        raise HTTPException(404, f"Backup '{filename}' nicht gefunden.")
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    return {"ok": True, "backup": backup_name, "config": new_cfg.model_dump()}


@app.post("/config/restart")
async def restart_service_endpoint():
    ok, message = config_editor.restart_service()
    if not ok:
        raise HTTPException(500, message)
    return {"ok": True, "message": message}


# --- OpenAI-kompatibler Proxy mit Auto-Load -------------------------------

HOP_BY_HOP = {"host", "authorization", "content-length", "connection"}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request):
    cfg = get_config()
    body = await request.body()

    if path == "models" and request.method == "GET":
        cached = list_cached_models(cfg.hf_home)
        names = sorted(set(cfg.models.keys()) | set(cached))
        return {
            "object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "vllm-manager"} for n in names],
        }

    model = None
    is_stream = False
    if body:
        try:
            parsed_body = json.loads(body)
            model = parsed_body.get("model")
            is_stream = bool(parsed_body.get("stream"))
        except Exception:
            pass
    model = model or cfg.default_model
    if not model:
        raise HTTPException(400, "Kein 'model' im Request-Body und kein default_model in config.json konfiguriert.")

    rid = telemetry.start_request(model, path, is_stream)

    cached = set(list_cached_models(cfg.hf_home))
    if model not in cfg.models and model not in cached:
        telemetry.finish_request(rid, "error")
        raise HTTPException(
            404,
            f"Modell '{model}' ist unbekannt. Erst per POST /models/pull herunterladen "
            f"oder in config.json unter \"models\" registrieren.",
        )

    try:
        engine_status = await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        telemetry.finish_request(rid, "error")
        raise HTTPException(503, str(e))
    telemetry.mark_ready(rid)

    target = f"http://{cfg.engine_host}:{engine_status['port']}/v1/{path}"
    client = httpx.AsyncClient(timeout=None)
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    req = client.build_request(request.method, target, content=body, headers=fwd_headers)
    upstream = await client.send(req, stream=True)
    result_status = "ok" if upstream.status_code < 400 else "error"

    async def gen():
        first_chunk = True
        sse_buffer = b""
        full_body = bytearray()
        prompt_tokens = None
        completion_tokens = None
        status = result_status
        try:
            async for chunk in upstream.aiter_raw():
                if first_chunk:
                    telemetry.mark_first_token(rid)
                    first_chunk = False
                if is_stream:
                    sse_buffer += chunk
                    while b"\n\n" in sse_buffer:
                        event, sse_buffer = sse_buffer.split(b"\n\n", 1)
                        for line in event.split(b"\n"):
                            if not line.startswith(b"data: "):
                                continue
                            payload = line[len(b"data: "):].strip()
                            if payload == b"[DONE]":
                                continue
                            try:
                                obj = json.loads(payload)
                            except Exception:
                                continue
                            choices = obj.get("choices") or []
                            if choices and (choices[0].get("delta") or {}).get("content"):
                                telemetry.increment_tokens(rid)
                            usage = obj.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                else:
                    full_body.extend(chunk)
                yield chunk
            if not is_stream and full_body:
                try:
                    obj = json.loads(bytes(full_body))
                    usage = obj.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                except Exception:
                    pass
        except Exception:
            status = "error"
            raise
        finally:
            telemetry.finish_request(rid, status, prompt_tokens, completion_tokens)
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
