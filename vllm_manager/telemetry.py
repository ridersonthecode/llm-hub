"""Live-Request-Tracking (für das Dashboard) + Parser für vLLMs eigene
Prometheus-Metriken (TTFT, TPOT, KV-Cache-Auslastung, Queue-Länge, ...)."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import httpx

from .config import get_config

MAX_RECENT = 30

active_requests: dict[str, dict] = {}
recent_requests: deque[dict] = deque(maxlen=MAX_RECENT)
last_request_at: Optional[float] = None
_request_counter = 0
_subscribers: list[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    if q in _subscribers:
        _subscribers.remove(q)


def _publish(event: dict) -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Dashboard bekommt beim nächsten Heartbeat ohnehin einen frischen Snapshot


def start_request(model: str, path: str) -> str:
    global last_request_at, _request_counter
    _request_counter += 1
    rid = f"{int(time.time() * 1000)}-{_request_counter}"
    last_request_at = time.time()
    active_requests[rid] = {
        "id": rid,
        "model": model,
        "path": path,
        "started_at": last_request_at,
        "ready_at": None,  # gesetzt sobald das Modell geladen ist und die Anfrage weitergereicht wird
        "queued_ms": None,  # Wartezeit auf Modell-Autostart/-Wechsel, falls nötig
        "ttft_ms": None,  # Zeit bis zum ersten Token AB ready_at (reine Generierungs-TTFT)
        "tokens_streamed": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "status": "running",
    }
    _publish({"type": "request_start"})
    return rid


def mark_ready(rid: str) -> None:
    """Modell ist geladen, Anfrage wird jetzt an die Engine weitergereicht."""
    r = active_requests.get(rid)
    if r is not None and r["ready_at"] is None:
        r["ready_at"] = time.time()
        r["queued_ms"] = round((r["ready_at"] - r["started_at"]) * 1000, 1)
        _publish({"type": "ready"})


def mark_first_token(rid: str) -> None:
    r = active_requests.get(rid)
    if r is not None and r["ttft_ms"] is None:
        base = r["ready_at"] or r["started_at"]
        r["ttft_ms"] = round((time.time() - base) * 1000, 1)
        _publish({"type": "first_token"})


def increment_tokens(rid: str) -> None:
    r = active_requests.get(rid)
    if r is not None:
        r["tokens_streamed"] += 1


def finish_request(
    rid: str,
    status: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    r = active_requests.pop(rid, None)
    if r is None:
        return
    r["status"] = status
    r["finished_at"] = time.time()
    r["duration_ms"] = round((r["finished_at"] - r["started_at"]) * 1000, 1)
    if prompt_tokens is not None:
        r["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        r["completion_tokens"] = completion_tokens
    recent_requests.appendleft(r)
    _publish({"type": "request_end"})


async def fetch_engine_metrics() -> dict:
    """Ruft vLLMs eigenen /metrics-Endpoint ab und extrahiert die wichtigsten Werte."""
    cfg = get_config()
    url = f"http://{cfg.engine_host}:{cfg.engine_port}/metrics"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(url)
            r.raise_for_status()
            return _parse_prometheus(r.text)
    except Exception:
        return {}


def _parse_prometheus(text: str) -> dict:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                _, value_str = rest.rsplit("}", 1)
            else:
                name, value_str = line.rsplit(" ", 1)
            value = float(value_str.strip())
        except ValueError:
            continue
        if name.startswith("vllm:"):
            # Bei mehreren Engines/Labels aufsummieren (bei uns i.d.R. nur eine aktiv)
            values[name] = values.get(name, 0.0) + value

    def avg_ms(sum_key: str, count_key: str) -> Optional[float]:
        count = values.get(count_key, 0.0)
        if not count:
            return None
        return round((values.get(sum_key, 0.0) / count) * 1000, 2)

    return {
        "num_requests_running": values.get("vllm:num_requests_running"),
        "num_requests_waiting": values.get("vllm:num_requests_waiting"),
        "kv_cache_usage_perc": values.get("vllm:kv_cache_usage_perc"),
        "prompt_tokens_total": values.get("vllm:prompt_tokens_total"),
        "generation_tokens_total": values.get("vllm:generation_tokens_total"),
        "avg_ttft_ms": avg_ms("vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count"),
        "avg_tpot_ms": avg_ms(
            "vllm:request_time_per_output_token_seconds_sum",
            "vllm:request_time_per_output_token_seconds_count",
        ),
        "avg_e2e_latency_ms": avg_ms("vllm:e2e_request_latency_seconds_sum", "vllm:e2e_request_latency_seconds_count"),
    }
