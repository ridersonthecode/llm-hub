"""MCP-Server (Streamable HTTP), damit eine KI den vLLM-Manager übers Netzwerk
steuern kann: Modelle auflisten, herunterladen, laden/entladen, Status abfragen."""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import downloader, process_manager
from .catalog import list_cached_models
from .config import get_config

mcp = FastMCP(
    "vllm-manager",
    # Ohne dies registriert FastMCP seine Route selbst unter "/mcp"; gemountet
    # unter "/mcp" ergäbe das intern "/mcp/mcp" -> 404. Mit "/" hier + Mount auf
    # "/mcp" in main.py landet die Route korrekt auf "/mcp/".
    streamable_http_path="/",
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
    cached = list_cached_models(cfg.hf_home)
    return {
        "registered": [
            {"model": name, "enabled": m.enabled, "notes": m.notes}
            for name, m in cfg.models.items()
        ],
        "cached_locally": cached,
        "currently_loaded": process_manager.engine.model,
        "default_model": cfg.default_model,
    }


@mcp.tool()
async def server_status() -> dict:
    """Gibt den Status der aktiven vLLM-Engine zurück (geladenes Modell, Laufzeit, Log-Datei)."""
    return process_manager.engine.status()


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
async def unload_model() -> dict:
    """Entlädt das aktuell geladene Modell und gibt den GPU-/Unified-Memory-Speicher frei."""
    await process_manager.stop_engine()
    return {"status": "unloaded"}
