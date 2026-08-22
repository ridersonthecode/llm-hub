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

from . import downloader, process_manager, telemetry
from .auth import require_api_key
from .catalog import list_cached_models
from .config import get_config, load_config
from .dashboard import router as dashboard_router
from .mcp_tools import mcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vllm_manager")

IDLE_CHECK_INTERVAL = 30


async def _idle_watchdog() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL)
        cfg = get_config()
        if cfg.idle_timeout_seconds and process_manager.engine.model:
            if time.time() - process_manager.engine.last_used > cfg.idle_timeout_seconds:
                logger.info(
                    "Idle-Timeout (%ss) erreicht, entlade %s",
                    cfg.idle_timeout_seconds,
                    process_manager.engine.model,
                )
                await process_manager.stop_engine(reason="idle_timeout")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_config()
    watchdog = asyncio.create_task(_idle_watchdog())
    async with mcp.session_manager.run():
        yield
    watchdog.cancel()
    await process_manager.stop_engine(reason="shutdown")


app = FastAPI(title="vLLM Manager", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())
app.include_router(dashboard_router)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    err = require_api_key(request)
    if err is not None:
        return err
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok", **process_manager.engine.status()}


@app.get("/models")
async def list_models_endpoint():
    cfg = get_config()
    cached = set(list_cached_models(cfg.hf_home))
    out = []
    for name, mcfg in cfg.models.items():
        out.append({
            "model": name,
            "cached": name in cached,
            "loaded": process_manager.engine.model == name,
            "enabled": mcfg.enabled,
            "notes": mcfg.notes,
        })
    known = {m["model"] for m in out}
    for name in sorted(cached - known):
        out.append({
            "model": name,
            "cached": True,
            "loaded": process_manager.engine.model == name,
            "enabled": True,
            "notes": "Lokal gecacht, aber nicht in config.json registriert.",
        })
    return {"models": out, "default_model": cfg.default_model, "active": process_manager.engine.model}


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
    if process_manager.engine.model == model:
        await process_manager.stop_engine()
        return {"status": "unloaded"}
    return {"status": "not_loaded", "currently_loaded": process_manager.engine.model}


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

    rid = telemetry.start_request(model, path)

    cached = set(list_cached_models(cfg.hf_home))
    if model not in cfg.models and model not in cached:
        telemetry.finish_request(rid, "error")
        raise HTTPException(
            404,
            f"Modell '{model}' ist unbekannt. Erst per POST /models/pull herunterladen "
            f"oder in config.json unter \"models\" registrieren.",
        )

    try:
        await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        telemetry.finish_request(rid, "error")
        raise HTTPException(503, str(e))
    telemetry.mark_ready(rid)

    target = f"http://{cfg.engine_host}:{cfg.engine_port}/v1/{path}"
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
