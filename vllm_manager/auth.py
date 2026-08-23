"""Optionale API-Key-Prüfung. Per config.json (api_key.enabled) an/aus schaltbar,
standardmäßig AUS - dann funktioniert der Server ohne jeden Auth-Header.

Reine ASGI-Middleware statt @app.middleware("http") (= Starlettes
BaseHTTPMiddleware): BaseHTTPMiddleware puffert/verzögert StreamingResponse-
Bodies, weil call_next() die Antwort intern über eine Queue/Background-Task
umleitet, statt den ASGI-`send`-Callable direkt durchzureichen. Das hat exakt
dieselbe Symptomatik verursacht wie der SSE-Flicker beim Dashboard (siehe
dashboard.py-Docstring) - nur diesmal am /v1-Chat-Proxy: gestreamte Antworten
(stream: true, von VS Code/Copilot genutzt) kamen beim Client nie an, weil
BaseHTTPMiddleware den Generator nicht inkrementell weiterreicht. Reines ASGI-
Middleware (ohne BaseHTTPMiddleware-Zwischenschicht) umgeht das vollständig."""
from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_config

EXEMPT_PATHS = {"/health", "/dashboard", "/dashboard/rag", "/dashboard/config"}


class ApiKeyMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # WebSockets (/dashboard/ws) und Lifespan-Events prüfen ihre eigene
            # Auth bzw. sind hiervon nicht betroffen.
            await self.app(scope, receive, send)
            return

        cfg = get_config()
        if not cfg.api_key.enabled or scope["path"] in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        if not cfg.api_key.key:
            response = JSONResponse(
                {"error": "api_key.enabled=true, aber api_key.key ist leer in config.json."},
                status_code=500,
            )
            await response(scope, receive, send)
            return

        headers = Headers(scope=scope)
        expected = f"Bearer {cfg.api_key.key}"
        if headers.get("authorization") != expected:
            response = JSONResponse(
                {"error": "Unauthorized. Gültigen 'Authorization: Bearer <key>' Header mitschicken."},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
