"""Local-only and anti-CSRF protections for the UI API.

* The server binds to 127.0.0.1 and nothing else (`check_bind`).
* The Host header must name 127.0.0.1 (blocks DNS-rebinding pages that resolve to it).
* Every POST, PUT, PATCH or DELETE needs the per-session token in `X-CSRF-Token` and an
  `Origin` from the local allowlist. Other sites cannot read the token (no CORS for them).
* Responses carry a strict Content-Security-Policy: scripts, styles, fonts and images load
  from this server only.
"""
from __future__ import annotations

import hmac
import secrets
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEV_PORT = 5173            # Vite dev server (proxies /api here)
CSRF_HEADER = "X-CSRF-Token"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; "
       "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class BindError(ValueError):
    pass


def check_bind(host: str) -> str:
    """The only address the API may listen on."""
    if host != LOCAL_HOST:
        raise BindError(f"the UI API binds to {LOCAL_HOST} only, not {host!r}")
    return host


def allowed_origins(port: int) -> frozenset[str]:
    return frozenset({f"http://{LOCAL_HOST}:{port}", f"http://{LOCAL_HOST}:{DEV_PORT}"})


def new_token() -> str:
    return secrets.token_urlsafe(32)


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    """Host check, CSRF check on unsafe methods, and security headers on every response."""

    def __init__(self, app: Any, *, token: str, origins: frozenset[str]):
        super().__init__(app)
        self.token, self.origins = token, origins

    async def dispatch(self, request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host != LOCAL_HOST:
            return _deny(400, "requests must be addressed to 127.0.0.1")
        if request.method in UNSAFE_METHODS:
            if request.headers.get("origin") not in self.origins:
                return _deny(403, "origin not allowed")
            sent = request.headers.get(CSRF_HEADER, "")
            if not sent or not hmac.compare_digest(sent, self.token):
                return _deny(403, "missing or invalid CSRF token")
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response


def _deny(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status,
                        headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"})


def scrub(obj: Any, secrets_: tuple[str, ...]) -> Any:
    """Replace any configured secret (the SEC User-Agent, API keys) wherever it appears."""
    live = tuple(s for s in secrets_ if s)
    if not live:
        return obj

    def walk(o: Any) -> Any:
        if isinstance(o, str):
            for s in live:
                o = o.replace(s, "[redacted]")
            return o
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o
    return walk(obj)
