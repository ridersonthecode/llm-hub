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

# Wie viele Zeichen von Reasoning/Content einer laufenden Anfrage fürs
# Live-Vorschau-Modal im Dashboard vorgehalten werden (siehe
# increment_tokens/increment_reasoning_tokens unten sowie dashboard.py,
# openPreviewModal) - nur der jeweils LETZTE Ausschnitt (Tail), kein
# unbegrenztes Wachstum bei langen Antworten.
PREVIEW_TAIL_CHARS = 4000

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
        "reasoning_tokens_streamed": 0,  # separat gezählt: Denkprozess-Chunks (delta.reasoning bzw. delta.reasoning_content, siehe reasoning_parser in config.json und main.py gen())
        # Live-Vorschau fürs Dashboard (siehe PREVIEW_TAIL_CHARS oben): NUR der
        # letzte Ausschnitt des bisher gestreamten Textes, damit man im
        # Active-Requests-Modal sieht, was während prefill/thinking/generating
        # tatsächlich passiert, statt nur den reinen Phasen-Namen.
        "content_preview": "",
        "reasoning_preview": "",
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


def mark_queued(rid: str) -> None:
    """Anfrage wartet in der globalen Concurrency-Warteschlange (siehe
    request_queue.py) auf einen freien Slot - noch VOR jedem Modell-Kaltstart/
    -Wechsel. Eigene Phase, damit das Dashboard "wartet auf Slot" von "Modell
    lädt gerade" (Phase "loading") unterscheiden kann."""
    _set_phase(rid, "queued")


def mark_dequeued(rid: str) -> None:
    """Slot wurde zugeteilt (siehe request_queue.acquire) - weiter mit dem
    normalen Ablauf (Modell laden/Anfrage weiterreichen), siehe mark_ready()."""
    _set_phase(rid, "loading")


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


def increment_tokens(rid: str, text: str = "") -> None:
    r = active_requests.get(rid)
    if r is not None:
        r["tokens_streamed"] += 1
        if text:
            r["content_preview"] = (r.get("content_preview", "") + text)[-PREVIEW_TAIL_CHARS:]
    _set_phase(rid, "generating")


def increment_reasoning_tokens(rid: str, text: str = "") -> None:
    r = active_requests.get(rid)
    if r is not None:
        r["reasoning_tokens_streamed"] += 1
        if text:
            r["reasoning_preview"] = (r.get("reasoning_preview", "") + text)[-PREVIEW_TAIL_CHARS:]
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

# (port, Metrik-Name) -> (letzter kumulativer sum-Stand, letzter count-Stand)
# bzw. -> letzter daraus berechneter "seit letzter Änderung"-Mittelwert in ms.
# Siehe _recent_avg_ms() - derselbe Differenz-über-Zeit-Trick wie
# system_metrics._read_cpu() (dort: CPU-Ticks aus /proc/stat), nur hier auf
# vLLMs eigene kumulative Latenz-Histogramme angewendet.
_prev_hist_samples: dict[tuple[int, str], tuple[float, float]] = {}
_recent_avg_ms: dict[tuple[int, str], Optional[float]] = {}


def _recent_avg(port: int, metric: str, cur_sum: float, cur_count: float) -> Optional[float]:
    """Mittelwert (in ms) NUR über die Anfragen, die seit der letzten Messung
    NEU abgeschlossen wurden - bewusst NICHT das Lifetime-Mittel seit Engine-
    Start, das vLLMs rohe Prometheus-Zähler eigentlich hergeben (sum/count
    sind monoton wachsende Zähler, die nie zurückgesetzt werden außer bei
    einem Engine-Neustart).

    Live beobachtet (2026-08-25): eine einzelne langsame historische Anfrage
    (z.B. während hoher Nebenlast, oder ein Kaltstart-Sample kurz nach dem
    Neuladen) zog den angezeigten Wert für den GESAMTEN Rest der Laufzeit
    dieser Engine dauerhaft nach unten - ein Nutzer, der live im Editor einen
    deutlich schnelleren Stream sah, hätte im Dashboard trotzdem dauerhaft
    einen viel schlechteren Wert gesehen ("1.5 Tok/s" trotz tatsächlich
    normaler Geschwindigkeit). Mit dieser Differenzbildung "vergisst" der
    angezeigte Wert alte, längst abgeschlossene Anfragen von selbst - er
    zeigt immer nur, was seit dem letzten Poll (per Dashboard-Heartbeat,
    ca. 1x/s) tatsächlich NEU abgeschlossen wurde.

    Gibt None zurück, solange nach einem (Neu-)Start noch keine einzige
    Anfrage abgeschlossen wurde. Danach bleibt der zuletzt berechnete Wert
    stehen, bis die nächste Anfrage fertig ist - kein Zittern zwischen "kein
    Wert" und einer Zahl, nur weil zwischen zwei Polls zufällig nichts fertig
    wurde."""
    key = (port, metric)
    prev_sum, prev_count = _prev_hist_samples.get(key, (0.0, 0.0))
    if cur_count < prev_count:
        # Zähler kleiner als beim letzten Mal -> Engine wurde neu gestartet
        # (frischer Prozess, frische Prometheus-Zähler) - alte Vergleichsbasis
        # verwerfen, sonst würde die nächste Differenz absurd groß/negativ.
        prev_sum, prev_count = 0.0, 0.0
    if cur_count > prev_count:
        d_sum = cur_sum - prev_sum
        d_count = cur_count - prev_count
        _recent_avg_ms[key] = round((d_sum / d_count) * 1000, 2)
        _prev_hist_samples[key] = (cur_sum, cur_count)
    return _recent_avg_ms.get(key)


def _get_metrics_client() -> httpx.AsyncClient:
    """Ein wiederverwendeter Client statt einem neuen pro Aufruf - vermeidet
    unnötigen Verbindungsaufbau bei jedem WS-Heartbeat/jeder Engine/jedem
    offenen Dashboard-Tab (Keep-Alive-Verbindung wird wiederverwendet)."""
    global _metrics_client
    if _metrics_client is None:
        _metrics_client = httpx.AsyncClient(timeout=2)
    return _metrics_client


async def fetch_engine_metrics(port: Optional[int] = None, engine: str = "vllm", model: Optional[str] = None) -> dict:
    """Ruft den /metrics-Endpoint der Engine ab und extrahiert die wichtigsten
    Werte. `port` adressiert eine bestimmte Engine aus dem Hot Pool - ohne
    Angabe wird der konfigurierte Default-Port verwendet. `engine` steuert,
    welches Prometheus-Namensschema geparst wird ("vllm" oder "llamacpp",
    siehe ModelConfig.engine) - beide Engines exponieren /metrics, aber mit
    unterschiedlichen Metrik-Namen/-Präfixen (vllm:... bzw. llamacpp:...).
    `model` wird nur für den llama.cpp-TTFT-Fallback gebraucht (siehe unten).
    Kurz gecacht pro Port (mit Lock-Dedup wie catalog.py) - mehrere
    gleichzeitig offene Dashboard-Tabs lösen so nur EINEN echten Abruf pro
    Engine/Sekunde aus, nicht einen pro Tab."""
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
            if engine == "llamacpp":
                result = _parse_prometheus_llamacpp(r.text)
                # llama-server hat kein TTFT-Histogramm (siehe
                # _parse_prometheus_llamacpp-Docstring) - Fallback auf die
                # eigene, engine-unabhängige Proxy-Messung (mark_first_token)
                # über die zuletzt abgeschlossenen Anfragen dieses Modells.
                result["avg_ttft_ms"] = _avg_ttft_from_recent_requests(model)
                result["avg_tpot_ms"] = _recent_avg(
                    resolved_port, "tpot", result.pop("_raw_tpot_sum"), result.pop("_raw_tpot_count")
                )
                result["avg_e2e_latency_ms"] = None
                result["kv_cache_usage_perc"] = await _fetch_llamacpp_kv_usage(resolved_port)
            else:
                result = _parse_prometheus(r.text)
                # Rohe kumulative Zähler in "seit letztem Poll neu
                # abgeschlossen" umrechnen (siehe _recent_avg()-Docstring)
                # statt des irreführenden Lifetime-Mittels seit Engine-Start.
                result["avg_ttft_ms"] = _recent_avg(
                    resolved_port, "ttft", result.pop("_raw_ttft_sum"), result.pop("_raw_ttft_count")
                )
                result["avg_tpot_ms"] = _recent_avg(
                    resolved_port, "tpot", result.pop("_raw_tpot_sum"), result.pop("_raw_tpot_count")
                )
                result["avg_e2e_latency_ms"] = _recent_avg(
                    resolved_port, "e2e", result.pop("_raw_e2e_sum"), result.pop("_raw_e2e_count")
                )
        except Exception:
            result = {}
        _metrics_cache[resolved_port] = result
        _metrics_cache_at[resolved_port] = time.time()
        return result


async def _fetch_llamacpp_kv_usage(port: int) -> Optional[float]:
    """KV-Cache-Auslastung als Bruch (0..1) für llama.cpp: kein fertiges Aggregat
    wie vLLMs kv_cache_usage_perc über /metrics (siehe
    _parse_prometheus_llamacpp-Docstring) - stattdessen selbst aus /slots
    (braucht das --slots-Flag, siehe ModelConfig.engine=="llamacpp"-
    extra_args) berechnet: n_prompt_tokens (Tokenanzahl des jeweils letzten
    Tasks pro Slot - bleibt laut llama.cpp server_slot::to_json() auch nach
    Abschluss stehen, solange kein neuer Task denselben Slot übernimmt) zu
    n_ctx (Kapazität des Slots) aufsummiert über alle Slots. Bei --parallel 1
    (unser Standardfall) ist das genau ein Slot, die Quote entspricht dann
    schlicht "aktuelle Konversationslänge / --ctx-size". Eigener Request statt
    Teil von /metrics - kein Problem, das Ergebnis landet ohnehin zusammen mit
    dem Rest im selben, kurz gecachten fetch_engine_metrics()-Aufruf. None,
    wenn der Endpoint fehlt/deaktiviert ist (kein --slots) oder kein Slot
    bisher einen Task gesehen hat."""
    cfg = get_config()
    url = f"http://{cfg.engine_host}:{port}/slots"
    try:
        r = await _get_metrics_client().get(url)
        r.raise_for_status()
        slots = r.json()
    except Exception:
        return None
    total_ctx = 0
    total_used = 0
    for slot in slots if isinstance(slots, list) else []:
        n_ctx = slot.get("n_ctx") or 0
        if n_ctx <= 0:
            continue
        total_ctx += n_ctx
        total_used += slot.get("n_prompt_tokens") or 0
    if total_ctx <= 0:
        return None
    # Als Bruch (0..1) zurückgeben, nicht als fertigen Prozentwert - wie
    # vLLMs kv_cache_usage_perc (siehe _parse_prometheus) erwartet das
    # Dashboard (fmtPct()) einen Bruch und multipliziert selbst mit 100.
    # Frueher stand hier bereits "* 100" -> fmtPct multiplizierte ein
    # zweites Mal, das Dashboard zeigte total unrealistische Werte wie
    # "4000%" statt z.B. "40%".
    return round(total_used / total_ctx, 4)


def _avg_ttft_from_recent_requests(model: Optional[str], max_samples: int = 20) -> Optional[float]:
    """TTFT-Fallback für Engines ohne eigenes TTFT-Histogramm (aktuell nur
    llama.cpp, siehe fetch_engine_metrics): Mittelwert über die ttft_ms-Werte
    (siehe mark_first_token) der letzten `max_samples` abgeschlossenen
    Anfragen GENAU dieses Modells aus recent_requests - misst der Proxy
    selbst, unabhängig davon, was die Engine an /metrics hergibt. Gleiches
    "nur die letzten, nicht das Lifetime-Mittel"-Prinzip wie _recent_avg()."""
    if model is None:
        return None
    samples = [
        r["ttft_ms"] for r in recent_requests
        if r.get("model") == model and r.get("ttft_ms") is not None
    ][:max_samples]
    if not samples:
        return None
    return round(sum(samples) / len(samples), 1)


def _parse_prometheus_values(text: str, prefix: str) -> dict[str, float]:
    """Rohes Prometheus-Text-Format (Name/Wert-Zeilen, "#"-Kommentare) auf ein
    dict Metrikname->Wert reduziert - gemeinsamer Kern für _parse_prometheus
    (vLLM, Präfix "vllm:") und _parse_prometheus_llamacpp (Präfix
    "llamacpp:"). Mehrere Zeilen mit gleichem Namen (z.B. Label-Varianten wie
    llama.cpp's spec_decode_..._per_pos_total{position="N"}) werden
    aufsummiert - bei uns i.d.R. sowieso nur eine Engine/ein Label aktiv."""
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
        if name.startswith(prefix):
            values[name] = values.get(name, 0.0) + value
    return values


def _parse_prometheus(text: str) -> dict:
    values = _parse_prometheus_values(text, "vllm:")
    return {
        "num_requests_running": values.get("vllm:num_requests_running"),
        "num_requests_waiting": values.get("vllm:num_requests_waiting"),
        "kv_cache_usage_perc": values.get("vllm:kv_cache_usage_perc"),
        "prompt_tokens_total": values.get("vllm:prompt_tokens_total"),
        "generation_tokens_total": values.get("vllm:generation_tokens_total"),
        # Rohe kumulative sum/count der drei Latenz-Histogramme - NICHT direkt
        # als Lifetime-Mittel anzeigen (siehe fetch_engine_metrics/_recent_avg_ms
        # für den Grund). "_raw_"-Präfix markiert: nur intern für die Delta-
        # Berechnung gedacht, wird dort wieder rausgenommen.
        "_raw_ttft_sum": values.get("vllm:time_to_first_token_seconds_sum", 0.0),
        "_raw_ttft_count": values.get("vllm:time_to_first_token_seconds_count", 0.0),
        "_raw_tpot_sum": values.get("vllm:request_time_per_output_token_seconds_sum", 0.0),
        "_raw_tpot_count": values.get("vllm:request_time_per_output_token_seconds_count", 0.0),
        "_raw_e2e_sum": values.get("vllm:e2e_request_latency_seconds_sum", 0.0),
        "_raw_e2e_count": values.get("vllm:e2e_request_latency_seconds_count", 0.0),
    }


def _parse_prometheus_llamacpp(text: str) -> dict:
    """Gegenstück zu _parse_prometheus für llama-server (--metrics-Flag
    nötig, siehe ModelConfig.engine=="llamacpp"-Docstring/extra_args) -
    eigenes, viel schmaleres Namensschema als vLLM (siehe llama.cpp,
    tools/server/server-task.cpp::to_metrics()). Deckungsgleich verfügbar:
    requests_processing/-deferred (~ vLLMs running/waiting),
    prompt_tokens_total, tokens_predicted_total (~ vLLMs generation_tokens_
    total) sowie tokens_predicted_seconds_total (Summe der reinen
    Decode-Zeit über alle generierten Tokens - Pendant zu vLLMs "time per
    output token"-Histogramm, hier aber als zwei einzelne Counter statt
    einem sum/count-Histogrammpaar). KEIN Pendant vorhanden für vLLMs TTFT-
    Histogramm und KV-Cache-Auslastung in Prozent (llama.cpp exponiert dafür
    nur Rohgrößen pro Slot über /slots, keine fertige Aggregat-Quote) - siehe
    fetch_engine_metrics für den TTFT-Fallback über die eigene Proxy-Messung;
    kv_cache_usage_perc bleibt bewusst None (bei "–" im Dashboard belassen,
    statt eine irreführende Kennzahl zu erfinden)."""
    values = _parse_prometheus_values(text, "llamacpp:")
    return {
        "num_requests_running": values.get("llamacpp:requests_processing"),
        "num_requests_waiting": values.get("llamacpp:requests_deferred"),
        "kv_cache_usage_perc": None,
        "prompt_tokens_total": values.get("llamacpp:prompt_tokens_total"),
        "generation_tokens_total": values.get("llamacpp:tokens_predicted_total"),
        # Siehe _raw_ttft_sum/-count-Kommentar in _parse_prometheus: gleiches
        # Prinzip, hier aber sum(Sekunden)/count(Tokens) statt sum/count(Anfragen)
        # - _recent_avg() ist dafür bewusst einheitenagnostisch.
        "_raw_tpot_sum": values.get("llamacpp:tokens_predicted_seconds_total", 0.0),
        "_raw_tpot_count": values.get("llamacpp:tokens_predicted_total", 0.0),
    }
