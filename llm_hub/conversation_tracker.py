"""Persistiert den tatsächlichen Inhalt abgeschlossener LLM-Anfragen (System-
Prompt, Nachrichtenverlauf, generierte Antwort inkl. Denkprozess) - Grundlage
für /dashboard/conversations. Ergänzt cost_tracker.py (das nur Metadaten wie
Tokens/Kosten/Dauer speichert, NIE den eigentlichen Text) um genau diesen
Inhalt, als eigene Datei, damit costs.jsonl schlank bleibt und beide
unabhängig voneinander gelöscht/zurückgesetzt werden können.

Persistiert als JSONL neben config.json (gleiches Muster wie costs.jsonl,
siehe cost_tracker.py-Docstring), damit die Historie einen Prozess-Neustart
übersteht.

Nur Anfragen, die tatsächlich an eine Engine weitergereicht wurden, landen
hier (siehe main.py _finish_and_record()/ollama_compat.py api_chat() -
Aufrufer übergeben request_messages/request_prompt nur dann). Ein früh
abgelehnter Request (unbekanntes Modell, Engine-Start fehlgeschlagen, noch in
der Warteschlange abgebrochen) hat keinen echten "Konversationsinhalt" - der
bleibt wie bisher nur als Metadaten-Zeile in cost_tracker sichtbar."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from .config import CONFIG_PATH

CONVERSATIONS_PATH = CONFIG_PATH.parent / "conversations.jsonl"

records: list[dict] = []
_counter = 0
_loaded = False
_subscribers: list[asyncio.Queue] = []

_PREVIEW_MAX_LEN = 160


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
    if CONVERSATIONS_PATH.exists():
        with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
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
    CONVERSATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONVERSATIONS_PATH.with_suffix(CONVERSATIONS_PATH.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(CONVERSATIONS_PATH)


def _build_preview(request_messages: Optional[list], request_prompt: Optional[str]) -> str:
    """Kurzer Ein-Zeilen-Text für die Tabellenspalte - die letzte user-Message
    (das ist üblicherweise die eigentliche Frage, nicht der oft viel längere
    System-Prompt) bzw. bei /v1/completions (kein messages-Array) der rohe
    Prompt. Whitespace/Zeilenumbrüche kollabiert, damit die Tabellenzeile
    nicht durch einen mehrzeiligen Prompt aufgerissen wird - der volle Text
    bleibt über das Modal einsehbar."""
    text = ""
    if request_messages:
        for m in reversed(request_messages):
            if m.get("role") == "user":
                content = m.get("content")
                # content kann bei Vision-Requests eine Liste aus Text-/Bild-
                # Teilen statt eines simplen Strings sein (OpenAI-Format).
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                break
    elif isinstance(request_prompt, str):
        text = request_prompt
    elif isinstance(request_prompt, list) and request_prompt:
        text = str(request_prompt[0])
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _PREVIEW_MAX_LEN:
        text = text[:_PREVIEW_MAX_LEN].rstrip() + "…"
    return text


def record_conversation(
    *,
    rid: str,
    model: str,
    path: str,
    started_at: float,
    finished_at: float,
    status: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    user_agent: Optional[str],
    request_messages: Optional[list] = None,
    request_prompt: Optional[str] = None,
    request_params: Optional[dict] = None,
    output_content: str = "",
    output_reasoning: str = "",
    finish_reason: Optional[str] = None,
) -> dict:
    _ensure_loaded()
    global _counter
    _counter += 1
    rec = {
        "id": f"{int(finished_at * 1000)}-{_counter}",
        "rid": rid,
        "model": model,
        "path": path,
        "user_agent": user_agent,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": round((finished_at - started_at) * 1000, 1),
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "messages": request_messages,
        "prompt": request_prompt,
        "request_params": request_params or {},
        "output_content": output_content,
        "output_reasoning": output_reasoning,
        "preview": _build_preview(request_messages, request_prompt),
    }
    records.append(rec)
    CONVERSATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONVERSATIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _publish()
    return rec


def list_records() -> list[dict]:
    _ensure_loaded()
    return list(reversed(records))  # neueste zuerst


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
