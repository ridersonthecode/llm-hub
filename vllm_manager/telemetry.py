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


def start_request(model: str, path: str, is_stream: bool = False, user_agent: Optional[str] = None) -> str:
    global last_request_at, _request_counter
    _request_counter += 1
    rid = f"{int(time.time() * 1000)}-{_request_counter}"
    last_request_at = time.time()
    active_requests[rid] = {
        "id": rid,
        "model": model,
        "path": path,
        "is_stream": is_stream,
        # Roher User-Agent-Header des aufrufenden Clients ("welche App hat den
        # Request geschickt") - None, falls der Client keinen mitschickt; die
        # Anzeige "unbekannt"/"unknown" übernimmt das Frontend (i18n), damit es
        # in beiden Sprachen passt statt hier fest verdrahtet zu sein.
        "user_agent": user_agent,
        "started_at": last_request_at,
        "ready_at": None,  # gesetzt sobald das Modell geladen ist und die Anfrage weitergereicht wird
        "queued_ms": None,  # Wartezeit auf Modell-Autostart/-Wechsel, falls nötig
        "ttft_ms": None,  # Zeit bis zum ersten Token AB ready_at (reine Generierungs-TTFT)
        "tokens_streamed": 0,
        "reasoning_tokens_streamed": 0,  # separat gezählt: Denkprozess-Chunks (delta.reasoning_content, siehe reasoning_parser in config.json)
        "prompt_tokens": None,
        "completion_tokens": None,
        "status": "running",
        # Phasen-Tracking fürs Dashboard ("was macht das Modell gerade"):
        # loading (Kaltstart/Modell wird geladen) -> prefill (Modell bereit,
        # Prompt wird verarbeitet, noch kein Output) -> thinking (Reasoning-
        # Content, siehe reasoning_parser) / tool_call / generating (normale
        # Antwort). phase_history sammelt jeden Wechsel mit Zeitstempel, damit
        # das Dashboard eine kleine Zeitleiste zeigen kann.
        "phase": "loading",
        "phase_history": [{"phase": "loading", "at": last_request_at}],
        # Gesetzt von mark_rag_used(), falls automatisches server-seitiges RAG
        # gegriffen hat (siehe rag.apply_auto_rag / ModelConfig.rag_collection).
        "rag_used": False,
        "rag_collection": None,
        "rag_hits": 0,
    }
    _publish({"type": "request_start"})
    return rid


def _set_phase(rid: str, phase: str) -> None:
    r = active_requests.get(rid)
    if r is not None and r.get("phase") != phase:
        r["phase"] = phase
        r["phase_history"].append({"phase": phase, "at": time.time()})
        _publish({"type": "phase_change"})


def mark_ready(rid: str) -> None:
    """Modell ist geladen, Anfrage wird jetzt an die Engine weitergereicht."""
    r = active_requests.get(rid)
    if r is not None and r["ready_at"] is None:
        r["ready_at"] = time.time()
        r["queued_ms"] = round((r["ready_at"] - r["started_at"]) * 1000, 1)
        _set_phase(rid, "prefill")
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
    _set_phase(rid, "generating")


def increment_reasoning_tokens(rid: str) -> None:
    r = active_requests.get(rid)
    if r is not None:
        r["reasoning_tokens_streamed"] += 1
    _set_phase(rid, "thinking")


def mark_tool_call(rid: str) -> None:
    _set_phase(rid, "tool_call")


def mark_rag_used(rid: str, collection: str, hits: int) -> None:
    """Markiert, dass für diese Anfrage automatisch RAG-Kontext eingefügt
    wurde (siehe rag.apply_auto_rag, aufgerufen von main.py/ollama_compat.py)
    - landet dank finish_request() (verschiebt denselben Dict in
    recent_requests) automatisch auch in der Verlaufsansicht, nicht nur bei
    Active Requests."""
    r = active_requests.get(rid)
    if r is not None:
        r["rag_used"] = True
        r["rag_collection"] = collection
        r["rag_hits"] = hits


def update_partial_usage(rid: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    """Aktualisiert prompt_tokens/completion_tokens auf einer noch LAUFENDEN
    Anfrage, sobald vLLM ein usage-Feld mitten im Stream schickt (nur wenn der
    Client stream_options.include_usage gesetzt hat - sonst kommt usage erst
    im allerletzten Chunk, praktisch zeitgleich mit finish_request()). Macht
    die exakten Zahlen fürs Dashboard live sichtbar, statt nur nach Abschluss
    der Anfrage in Recent Requests."""
    r = active_requests.get(rid)
    if r is None:
        return
    if prompt_tokens is not None:
        r["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        r["completion_tokens"] = completion_tokens


def finish_request(
    rid: str,
    status: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Optional[dict]:
    """Gibt den fertigen Request-Datensatz zurück (None, falls rid unbekannt) -
    genutzt von main.py, um denselben Datensatz an cost_tracker.record_request()
    weiterzureichen, ohne started_at/finished_at doppelt zu berechnen."""
    r = active_requests.pop(rid, None)
    if r is None:
        return None
    r["status"] = status
    r["finished_at"] = time.time()
    r["duration_ms"] = round((r["finished_at"] - r["started_at"]) * 1000, 1)
    if prompt_tokens is not None:
        r["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        r["completion_tokens"] = completion_tokens
    recent_requests.appendleft(r)
    _publish({"type": "request_end"})
    return r


_metrics_client: Optional[httpx.AsyncClient] = None
_METRICS_CACHE_TTL = 0.8  # wie system_metrics.py - mehrere offene Dashboard-Tabs teilen sich einen Abruf
_metrics_cache: dict[int, dict] = {}
_metrics_cache_at: dict[int, float] = {}
_metrics_locks: dict[int, asyncio.Lock] = {}


def _get_metrics_client() -> httpx.AsyncClient:
    """Ein wiederverwendeter Client statt einem neuen pro Aufruf - vermeidet
    unnötigen Verbindungsaufbau bei jedem WS-Heartbeat/jeder Engine/jedem
    offenen Dashboard-Tab (Keep-Alive-Verbindung wird wiederverwendet)."""
    global _metrics_client
    if _metrics_client is None:
        _metrics_client = httpx.AsyncClient(timeout=2)
    return _metrics_client


async def fetch_engine_metrics(port: Optional[int] = None) -> dict:
    """Ruft vLLMs eigenen /metrics-Endpoint ab und extrahiert die wichtigsten
    Werte. `port` adressiert eine bestimmte Engine aus dem Hot Pool - ohne
    Angabe wird der konfigurierte Default-Port verwendet. Kurz gecacht pro
    Port (mit Lock-Dedup wie catalog.py) - mehrere gleichzeitig offene
    Dashboard-Tabs lösen so nur EINEN echten Abruf pro Engine/Sekunde aus,
    nicht einen pro Tab."""
    cfg = get_config()
    resolved_port = port or cfg.engine_port
    now = time.time()
    if resolved_port in _metrics_cache and (now - _metrics_cache_at.get(resolved_port, 0)) < _METRICS_CACHE_TTL:
        return _metrics_cache[resolved_port]

    lock = _metrics_locks.setdefault(resolved_port, asyncio.Lock())
    async with lock:
        # Zwischen dem ungelockten Check oben und hier könnte ein anderer Task
        # den Cache schon aufgefrischt haben - erneut prüfen.
        now = time.time()
        if resolved_port in _metrics_cache and (now - _metrics_cache_at.get(resolved_port, 0)) < _METRICS_CACHE_TTL:
            return _metrics_cache[resolved_port]
        url = f"http://{cfg.engine_host}:{resolved_port}/metrics"
        try:
            r = await _get_metrics_client().get(url)
            r.raise_for_status()
            result = _parse_prometheus(r.text)
        except Exception:
            result = {}
        _metrics_cache[resolved_port] = result
        _metrics_cache_at[resolved_port] = time.time()
        return result


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
