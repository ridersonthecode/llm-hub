"""MCP-Server (Streamable HTTP), damit eine KI den vLLM-Manager übers Netzwerk
steuern kann: Modelle auflisten, herunterladen, laden/entladen, Status abfragen."""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import downloader, process_manager, rag
from .catalog import list_cached_models
from .config import get_config

mcp = FastMCP(
    "vllm-manager",
    # Ohne dies registriert FastMCP seine Route selbst unter "/mcp"; gemountet
    # unter "/mcp" ergäbe das intern "/mcp/mcp" -> 404. Mit "/" hier + Mount auf
    # "/mcp" in main.py landet die Route korrekt auf "/mcp/".
    streamable_http_path="/",
    # FastMCP aktiviert DNS-Rebinding-Schutz automatisch, sobald `host`
    # (Default "127.0.0.1", von uns nie überschrieben - wir mounten die App ja
    # nur, statt FastMCPs eigenen Uvicorn zu nutzen) in ("127.0.0.1", "localhost",
    # "::1") liegt: dann werden nur noch Host-Header wie "127.0.0.1:*" akzeptiert.
    # Zugriffe übers LAN (z.B. "10.7.21.3:11434") bekommen dadurch 421 "Invalid
    # Host header". Da dieser Dienst bewusst netzwerkweit ohne Einschränkung
    # erreichbar sein soll (siehe Sicherheit in Anleitung.md), hier explizit aus.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    instructions=(
        "Verwaltet einen vLLM-Inferenzserver auf diesem Host: Modelle auflisten, "
        "per HuggingFace herunterladen (mit Fortschritts-Telemetrie), laden/entladen "
        "und den Serverstatus abfragen. Ein Modell muss erst lokal vorhanden sein "
        "(pull_model), bevor es geladen/genutzt werden kann - registrierte Modelle "
        "siehe list_models()."
    ),
)


@mcp.tool()
async def list_models() -> dict:
    """Listet registrierte, lokal gecachte und aktuell geladene Modelle."""
    cfg = get_config()
    cached = await list_cached_models(cfg.hf_home)
    return {
        "registered": [
            {"model": name, "enabled": m.enabled, "notes": m.notes}
            for name, m in cfg.models.items()
        ],
        "cached_locally": cached,
        "currently_loaded": process_manager.loaded_models(),
        "default_model": cfg.default_model,
    }


@mcp.tool()
async def server_status() -> dict:
    """Gibt den Status aller aktiven vLLM-Engines zurück (Hot Pool: bei
    max_concurrent_models > 1 können mehrere Modelle gleichzeitig geladen
    sein). Je Eintrag: geladenes Modell, Port, Status, Laufzeit, Log-Datei."""
    cfg = get_config()
    return {
        "engines": [e.status() for e in process_manager.engines.values()],
        "max_concurrent_models": cfg.max_concurrent_models,
    }


@mcp.tool()
async def pull_model(model: str, revision: Optional[str] = None, hf_token: Optional[str] = None) -> dict:
    """Startet den Download eines HuggingFace-Modells im Hintergrund.

    Gibt eine job_id zurück - Fortschritt mit pull_status(job_id) abfragen.
    """
    job_id = await downloader.start_job(model, revision, hf_token)
    return {"job_id": job_id}


@mcp.tool()
async def pull_status(job_id: str) -> dict:
    """Fragt Fortschritt eines Downloads ab: bytes_done/bytes_total, percent, speed_mbps, eta_seconds, state."""
    job = downloader.get_job(job_id)
    if job is None:
        return {"error": f"Unbekannte job_id: {job_id}"}
    return job


@mcp.tool()
async def load_model(model: str) -> dict:
    """Lädt ein Modell in die vLLM-Engine (startet sie bei Bedarf neu). Blockiert bis das Modell bereit ist."""
    return await process_manager.ensure_loaded(model)


@mcp.tool()
async def unload_model(model: Optional[str] = None) -> dict:
    """Entlädt ein Modell und gibt seinen Unified-Memory-Speicher frei.
    Ohne `model` werden ALLE aktuell geladenen Modelle entladen (z.B. bei
    einem Hot Pool mit mehreren gleichzeitig laufenden Modellen)."""
    await process_manager.stop_engine(model)
    return {"status": "unloaded", "model": model or "all"}


# --- RAG (Retrieval-Augmented Generation) ---------------------------------
# Muss in config.json unter "rag" aktiviert sein (embedding_model gesetzt),
# sonst werfen diese Tools einen Fehler mit Hinweis, was zu konfigurieren ist.

@mcp.tool()
async def rag_add_text(text: str, collection: Optional[str] = None, source: str = "text") -> dict:
    """Fügt einen Text der RAG-Wissensdatenbank hinzu (wird automatisch in
    Chunks zerlegt und eingebettet). `collection` gruppiert zusammengehörige
    Dokumente (z.B. pro Projekt) - ohne Angabe wird die Default-Collection aus
    config.json verwendet. Gibt document_id und Anzahl erzeugter Chunks zurück."""
    cfg = get_config()
    col = collection or cfg.rag.default_collection
    return await rag.add_text(col, text, source=source)


@mcp.tool()
async def rag_add_file(path: str, collection: Optional[str] = None) -> dict:
    """Fügt eine Datei (PDF oder Text/Markdown) der RAG-Wissensdatenbank hinzu.
    WICHTIG: `path` muss auf dem Server laufen, auf dem der vLLM-Manager läuft
    (nicht auf deinem eigenen Rechner) - für Dateien von woanders stattdessen
    den Inhalt lesen und rag_add_text() verwenden."""
    cfg = get_config()
    col = collection or cfg.rag.default_collection
    return await rag.add_file(col, path)


@mcp.tool()
async def rag_search(query: str, collection: Optional[str] = None, top_k: int = 5) -> dict:
    """Sucht die relevantesten Text-Ausschnitte zu einer Anfrage in der RAG-
    Wissensdatenbank (semantische Ähnlichkeitssuche, kein reiner Stichwort-
    Abgleich). Ergebnis absteigend nach Relevanz sortiert."""
    cfg = get_config()
    col = collection or cfg.rag.default_collection
    return {"collection": col, "results": await rag.search(col, query, top_k)}


@mcp.tool()
async def rag_list_collections() -> dict:
    """Listet alle RAG-Collections mit Anzahl gespeicherter Chunks."""
    return {"collections": await rag.list_collections()}


@mcp.tool()
async def rag_list_documents(collection: Optional[str] = None) -> dict:
    """Listet alle Dokumente (gruppiert aus den einzelnen Chunks) einer
    Collection, neueste zuerst."""
    cfg = get_config()
    col = collection or cfg.rag.default_collection
    return {"collection": col, "documents": await rag.list_documents(col)}


@mcp.tool()
async def rag_delete_document(document_id: str, collection: Optional[str] = None) -> dict:
    """Löscht ein Dokument (alle seine Chunks) aus einer Collection."""
    cfg = get_config()
    col = collection or cfg.rag.default_collection
    return await rag.delete_document(col, document_id)


@mcp.tool()
async def rag_delete_collection(collection: str) -> dict:
    """Löscht eine komplette RAG-Collection inkl. aller enthaltenen Dokumente."""
    return await rag.delete_collection(collection)

