"""Optionale API-Key-Prüfung. Per config.json (api_key.enabled) an/aus schaltbar,
standardmäßig AUS - dann funktioniert der Server ohne jeden Auth-Header."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import get_config

EXEMPT_PATHS = {"/health", "/dashboard"}


def require_api_key(request: Request) -> JSONResponse | None:
    cfg = get_config()
    if not cfg.api_key.enabled:
        return None
    if request.url.path in EXEMPT_PATHS:
        return None
    if not cfg.api_key.key:
        return JSONResponse(
            {"error": "api_key.enabled=true, aber api_key.key ist leer in config.json."},
            status_code=500,
        )
    expected = f"Bearer {cfg.api_key.key}"
    if request.headers.get("authorization") != expected:
        return JSONResponse(
            {"error": "Unauthorized. Gültigen 'Authorization: Bearer <key>' Header mitschicken."},
            status_code=401,
        )
    return None
