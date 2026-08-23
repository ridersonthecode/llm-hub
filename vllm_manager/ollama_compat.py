"""Ollama-Kompatibilitäts-Layer: übersetzt Ollamas natives POST /api/chat
(+ GET /api/tags) auf den OpenAI-kompatiblen vLLM-Proxy, damit ältere, für
Ollama gebaute Tools (z.B. eigene Skripte mit urllib/requests gegen
"{base_url}/api/chat") ohne Codeänderung weiterlaufen - nur die alten
Ollama-Modellnamen werden auf die neuen HuggingFace-Namen gemappt.

Deckt bewusst nur ab, was reale Alt-Clients hier tatsächlich nutzen (siehe
Anleitung.md, Abschnitt "Ollama-Kompatibilität"): chat mit optionalem
strukturiertem JSON-Output (`format`) und Thinking-Toggle (`think`). Andere
Ollama-Endpunkte (/api/generate, /api/pull, /api/show, ...) gibt es nicht -
bei Bedarf hier nach demselben Muster ergänzen."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import process_manager, rag, telemetry
from .catalog import list_cached_models
from .config import Config, get_config

logger = logging.getLogger("vllm_manager.ollama_compat")
router = APIRouter()

# Alte Ollama-Tags -> neue HuggingFace-Namen (siehe Anleitung.md, "Von Ollama
# auf vLLM übernommene Modelle"). Nur ein Fallback: wer direkt den neuen
# HF-Namen schickt, funktioniert genauso.
OLLAMA_MODEL_ALIASES: dict[str, str] = {
    "qwen3:8b": "Qwen/Qwen3-8B",
    "nemotron-3-nano:4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    "qwen3.8:27b": "Qwen/Qwen3.8-27B-FP8",
    "qwen3-coder:30b-a3b-q4_K_M": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "qwen3-coder-256k:latest": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "qwen3.6:35b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "nemotron-3.5-lightning:latest": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "nemotron-cascade-2:latest": "nvidia/Nemotron-Cascade-2-30B-A3B",
    "qwen3-next:80b": "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
    "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
}


def _resolve_model(name: str, cfg: Config, cached: set[str]) -> str:
    if name in cfg.models or name in cached:
        return name
    return OLLAMA_MODEL_ALIASES.get(name, name)


@router.get("/api/tags")
async def api_tags():
    """Ollama-kompatible Modell-Liste - manche Alt-Tools prüfen beim Start
    damit, welche Modelle überhaupt verfügbar sind. Enthält sowohl die neuen
    HF-Namen als auch die alten Ollama-Aliase, sofern deren Ziel verfügbar ist."""
    cfg = get_config()
    cached = set(await list_cached_models(cfg.hf_home))
    available = set(cfg.models.keys()) | cached
    names = set(available)
    for alias, target in OLLAMA_MODEL_ALIASES.items():
        if target in available:
            names.add(alias)
    names = sorted(names)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "models": [
            {"name": n, "model": n, "modified_at": now, "size": 0, "digest": "", "details": {}}
            for n in names
        ]
    }


@router.post("/api/chat")
async def api_chat(request: Request):
    cfg = get_config()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Ungültiges JSON im Request-Body.")

    if body.get("stream"):
        # Ollamas Streaming-Format ist NDJSON mit anderer Chunk-Struktur als
        # OpenAI-SSE - hier (noch) nicht übersetzt, um keine kaputten
        # Halb-Antworten auszuliefern. Alt-Clients mit stream:false (der
        # Ollama-Default) sind davon nicht betroffen.
        raise HTTPException(
            501,
            "Ollama-Kompatibilitäts-Layer unterstützt aktuell nur \"stream\": false.",
        )

    ollama_model = body.get("model")
    if not ollama_model:
        raise HTTPException(400, "'model' fehlt im Request-Body.")

    cached = set(await list_cached_models(cfg.hf_home))
    model = _resolve_model(ollama_model, cfg, cached)
    if model not in cfg.models and model not in cached:
        raise HTTPException(
            404,
            f"Modell '{ollama_model}' ist unbekannt (auch nicht als Ollama-Alias bekannt). "
            f"Erst per POST /models/pull herunterladen oder in config.json registrieren.",
        )

    openai_body: dict = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": False,
    }
    fmt = body.get("format")
    if isinstance(fmt, dict):
        # Ollamas "format": <json-schema> -> OpenAIs strukturiertes
        # response_format (von vLLM per Guided Decoding unterstützt).
        openai_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt},
        }
    elif fmt == "json":
        openai_body["response_format"] = {"type": "json_object"}
    if "think" in body:
        openai_body["chat_template_kwargs"] = {"enable_thinking": bool(body["think"])}
    options = body.get("options") or {}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("num_predict", "max_tokens")):
        if src in options:
            openai_body[dst] = options[src]

    # Sicherheitsnetz gegen durchgehende Generierungen (siehe main.py
    # _apply_default_max_tokens / config.py ModelConfig.max_tokens) - nur wenn
    # der Alt-Client (über num_predict) selbst nichts vorgegeben hat.
    mcfg = cfg.models.get(model)
    if mcfg is not None and mcfg.max_tokens is not None and "max_tokens" not in openai_body:
        openai_body["max_tokens"] = mcfg.max_tokens
    # Vorbeugung gegen Wiederholungsschleifen (siehe main.py
    # _apply_default_repetition_penalty / config.py ModelConfig.
    # repetition_penalty) - nur wenn der Alt-Client selbst nichts vorgibt.
    if mcfg is not None and mcfg.repetition_penalty is not None and "repetition_penalty" not in openai_body:
        openai_body["repetition_penalty"] = mcfg.repetition_penalty

    rid = telemetry.start_request(model, "api/chat", user_agent=request.headers.get("user-agent"))
    try:
        engine_status = await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        telemetry.finish_request(rid, "error")
        raise HTTPException(503, str(e))
    telemetry.mark_ready(rid)

    # Automatisches server-seitiges RAG (siehe rag.apply_auto_rag /
    # ModelConfig.rag_collection) - profitiert auch von Alt-Tools, die dieses
    # Ollama-kompatible /api/chat statt der OpenAI-API nutzen.
    rag_result = await rag.apply_auto_rag(model, openai_body.get("messages") or [])
    if rag_result:
        telemetry.mark_rag_used(rid, ", ".join(rag_result["collections"]), rag_result["hits"])

    target = f"http://{cfg.engine_host}:{engine_status['port']}/v1/chat/completions"
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.post(target, json=openai_body)
    except Exception as e:
        telemetry.finish_request(rid, "error")
        raise HTTPException(502, f"vLLM-Engine für '{model}' nicht erreichbar: {e}")

    if upstream.status_code >= 400:
        telemetry.finish_request(rid, "error")
        raise HTTPException(upstream.status_code, upstream.text)

    telemetry.mark_first_token(rid)
    data = upstream.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    usage = data.get("usage") or {}
    telemetry.finish_request(rid, "ok", usage.get("prompt_tokens"), usage.get("completion_tokens"))

    duration_ns = int((time.time() - started) * 1e9)
    return JSONResponse({
        "model": ollama_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "total_duration": duration_ns,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": duration_ns,
    })
