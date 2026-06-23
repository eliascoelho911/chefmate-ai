"""
Security middleware for Chefmate AI.

Enforces API key authentication on all incoming requests.
The healthcheck endpoint is exempt.
"""

import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates the X-API-Key header against CHEFMATE_API_KEY.
    Returns 401 for missing or invalid keys.
    """

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self._expected_key = api_key or os.getenv("CHEFMATE_API_KEY")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Healthcheck is always public
        if request.url.path == "/health":
            return await call_next(request)

        if not self._expected_key:
            return Response(
                content='{"detail":"Server misconfiguration: API key not set"}',
                status_code=500,
                media_type="application/json",
            )

        provided = request.headers.get("X-API-Key")
        if provided != self._expected_key:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
