"""Fiktives Kostentracking: rechnet abgeschlossene Anfragen gegen konfigurierbare
$/MTok-Preise um (Default: Claude-Sonnet-5-Standardpreise, siehe config.Pricing) -
rein zum Vergleich "was hätte das über eine Cloud-API gekostet". Der lokale
Betrieb bleibt selbstverständlich kostenlos, das hier beeinflusst nichts
Funktionales. Persistiert als JSONL neben config.json (gleiches Muster wie die
Backups in config_editor.py), damit die Historie einen Prozess-Neustart übersteht.

Kosten werden NUR berechnet, wenn prompt_tokens UND completion_tokens exakt
bekannt sind (Nicht-Streaming, oder Streaming mit stream_options.include_usage -
siehe Anleitung.md-Caveat zu Token-Zählung). Ohne das gibt es bewusst KEINE
Zahl statt einer geschätzten - siehe dashboard.py für die separate, klar als
Live-Schätzung markierte "so weit"-Anzeige bei noch laufenden Anfragen."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from .config import CONFIG_PATH, get_config

COSTS_PATH = CONFIG_PATH.parent / "costs.jsonl"

records: list[dict] = []
_counter = 0
_loaded = False
_subscribers: list[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    if q in _subscribers:
        _subscribers.remove(q)


def _publish() -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass  # nächster Heartbeat holt ohnehin einen frischen Snapshot


def _ensure_loaded() -> None:
    global _loaded, _counter
    if _loaded:
        return
    _loaded = True
    if COSTS_PATH.exists():
        with open(COSTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    _counter = len(records)


def _rewrite_file() -> None:
    COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COSTS_PATH.with_suffix(COSTS_PATH.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(COSTS_PATH)


def pricing_for(model: str) -> tuple[float, float]:
    """(input_per_mtok, output_per_mtok) - Modell-Override falls in config.json
    gesetzt (Config-Editor), sonst der globale default_pricing-Fallback."""
    cfg = get_config()
    mcfg = cfg.models.get(model)
    p = (mcfg.pricing if mcfg else None) or cfg.default_pricing
    return p.input_per_mtok, p.output_per_mtok


def compute_cost(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[dict]:
    if prompt_tokens is None or completion_tokens is None:
        return None
    in_rate, out_rate = pricing_for(model)
    input_cost = prompt_tokens / 1_000_000 * in_rate
    output_cost = completion_tokens / 1_000_000 * out_rate
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "cost_usd": round(input_cost + output_cost, 6),
        "pricing_input_per_mtok": in_rate,
        "pricing_output_per_mtok": out_rate,
    }


def record_request(
    model: str,
    path: str,
    started_at: float,
    finished_at: float,
    status: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    user_agent: Optional[str] = None,
) -> dict:
    """Persistiert eine abgeschlossene Anfrage - auch fehlgeschlagene (mit
    cost_usd=None), damit die Kostenseite wirklich JEDE Anfrage zeigt, nicht
    nur bepreiste."""
    _ensure_loaded()
    global _counter
    _counter += 1
    rec = {
        "id": f"{int(finished_at * 1000)}-{_counter}",
        "model": model,
        "path": path,
        "user_agent": user_agent,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": round((finished_at - started_at) * 1000, 1),
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "cost_usd": None,
        "pricing_input_per_mtok": None,
        "pricing_output_per_mtok": None,
    }
    cost = compute_cost(model, prompt_tokens, completion_tokens)
    if cost:
        rec.update(cost)
    records.append(rec)
    COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COSTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _publish()
    return rec


def list_records() -> list[dict]:
    _ensure_loaded()
    return list(reversed(records))  # neueste zuerst


def summary() -> dict:
    _ensure_loaded()
    priced = [r for r in records if r.get("cost_usd") is not None]
    total_cost = sum(r["cost_usd"] for r in priced)
    by_model: dict[str, dict] = {}
    for r in records:
        m = by_model.setdefault(
            r["model"],
            {"model": r["model"], "requests": 0, "priced_requests": 0, "cost_usd": 0.0},
        )
        m["requests"] += 1
        if r.get("cost_usd") is not None:
            m["priced_requests"] += 1
            m["cost_usd"] += r["cost_usd"]
    for m in by_model.values():
        m["cost_usd"] = round(m["cost_usd"], 6)
    return {
        "total_requests": len(records),
        "priced_requests": len(priced),
        "total_cost_usd": round(total_cost, 6),
        "by_model": sorted(by_model.values(), key=lambda m: m["cost_usd"], reverse=True),
    }


def delete_records(ids: list[str]) -> int:
    _ensure_loaded()
    id_set = set(ids)
    before = len(records)
    records[:] = [r for r in records if r["id"] not in id_set]
    removed = before - len(records)
    if removed:
        _rewrite_file()
        _publish()
    return removed


def reset_all() -> int:
    _ensure_loaded()
    removed = len(records)
    records[:] = []
    _rewrite_file()
    _publish()
    return removed
