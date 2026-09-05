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

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import active_streams, conversation_tracker, process_manager, rag, request_queue, telemetry
from .catalog import list_cached_models
from .config import Config, get_config

logger = logging.getLogger("llm_hub.ollama_compat")
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
    cached = set(await list_cached_models(cfg.resolved_hf_home()))
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

    cached = set(await list_cached_models(cfg.resolved_hf_home()))
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
    # Dieselbe globale Concurrency-Grenze wie proxy_v1() in main.py (siehe
    # request_queue.py) - gilt "egal an welches Modell", also auch für
    # Alt-Clients über dieses Ollama-Kompatibilitäts-Layer.
    await request_queue.acquire(rid)
    try:
        engine_status = await process_manager.ensure_loaded(model)
    except (RuntimeError, TimeoutError) as e:
        request_queue.release()
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
    # Bewusst NICHT mehr ein einziges blockierendes client.post() (wie vor
    # dem 2026-09-01-Fix) - das registrierte nirgends eine abbrechbare
    # Verbindung, weshalb der Dashboard-Cancel-Button hier immer mit "Keine
    # aktive, abbrechbare Anfrage gefunden" scheiterte, sobald der
    # aufrufende Alt-Client (z.B. Python mit eigenem Timeout) längst
    # aufgegeben hatte, die Engine aber unbemerkt weiter generierte. Wie
    # main.py proxy_v1()/gen(): client.send(..., stream=True) + manuelles
    # Einlesen, dazwischen in active_streams registriert - macht sowohl den
    # manuellen Cancel-Button als auch die automatische Disconnect-Erkennung
    # (siehe disconnect_watcher unten) für diesen Pfad nutzbar.
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("POST", target, json=openai_body)
    try:
        upstream = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        request_queue.release()
        telemetry.finish_request(rid, "error")
        raise HTTPException(502, f"vLLM-Engine für '{model}' nicht erreichbar: {e}")

    active_streams.register(rid, upstream)
    disconnect_watcher = asyncio.create_task(active_streams.watch_disconnect(request, rid))
    status = "ok"
    read_error: Optional[Exception] = None

    # Für /dashboard/conversations (conversation_tracker.py) - siehe dortigen
    # Docstring: nur ab hier geloggt, weil die Anfrage jetzt tatsächlich an die
    # Engine ging (die früheren Ablehnungen oben - unbekanntes Modell,
    # ensure_loaded()-Fehlschlag, Engine unerreichbar - haben keinen echten
    # "Konversationsinhalt"). openai_body["messages"] enthält bereits den
    # evtl. per Auto-RAG injizierten Kontext (siehe rag.apply_auto_rag oben).
    request_params = {k: v for k, v in openai_body.items() if k not in ("messages", "model")}

    def _record_conversation(fin_status: str, output_content: str = "", output_reasoning: str = "",
                              finish_reason: Optional[str] = None,
                              prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None) -> None:
        conversation_tracker.record_conversation(
            rid=rid, model=model, path="api/chat",
            started_at=started, finished_at=time.time(), status=fin_status,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            user_agent=request.headers.get("user-agent"),
            request_messages=openai_body.get("messages"), request_params=request_params,
            output_content=output_content, output_reasoning=output_reasoning, finish_reason=finish_reason,
        )
    try:
        full_body = bytearray()
        async for chunk in upstream.aiter_raw():
            full_body.extend(chunk)
    except Exception as e:
        # Verbindung zur Engine ist WÄHREND des Lesens abgebrochen - anders
        # als beim alten client.post() (ein einziger Aufruf, ein einziger
        # except-Block oben) kann das jetzt hier separat passieren, seit wir
        # den Stream selbst einlesen. was_cancelled() unterscheidet das von
        # unserem eigenen absichtlichen upstream.aclose() (Cancel-Button/
        # Disconnect-Watcher) - dort ist der Abbruch das gewünschte Ergebnis,
        # kein echter Fehler.
        if active_streams.was_cancelled(rid):
            status = "cancelled"
        else:
            status = "error"
            read_error = e
    finally:
        disconnect_watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_watcher
        active_streams.unregister(rid)
        await upstream.aclose()
        await client.aclose()

    if status == "cancelled":
        request_queue.release()
        telemetry.finish_request(rid, "cancelled")
        _record_conversation("cancelled")
        # Client ist per Definition nicht mehr da (manueller Cancel oder
        # erkannter Disconnect) - diese Response erreicht ihn ohnehin nicht
        # mehr, der Status-Code ist nur fürs Log/evtl. Proxies relevant.
        raise HTTPException(499, "Anfrage wurde abgebrochen (manuell oder Client-Disconnect erkannt).")
    if status == "error":
        request_queue.release()
        telemetry.finish_request(rid, "error")
        _record_conversation("error")
        raise HTTPException(502, f"vLLM-Engine für '{model}' nicht erreichbar: {read_error}")

    if upstream.status_code >= 400:
        request_queue.release()
        telemetry.finish_request(rid, "error")
        _record_conversation("error", output_content=bytes(full_body).decode("utf-8", "replace"))
        raise HTTPException(upstream.status_code, bytes(full_body).decode("utf-8", "replace"))

    telemetry.mark_first_token(rid)
    data = json.loads(bytes(full_body))
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    request_queue.release()
    telemetry.finish_request(rid, "ok", usage.get("prompt_tokens"), usage.get("completion_tokens"))
    _record_conversation(
        "ok", output_content=content, output_reasoning=reasoning, finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
    )

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
