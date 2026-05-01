from __future__ import annotations

import hmac
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from bbsia.app.runtime.audit import _audit_event
from bbsia.app.runtime.state import (
    ADMIN_API_KEY,
    ADMIN_PATHS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SEC,
    READ_API_KEY,
    _REQUEST_LOCK,
    _REQUEST_LOG,
)


def _configured_read_keys() -> list[str]:
    return [key for key in (READ_API_KEY, ADMIN_API_KEY) if key]


def _configured_admin_keys() -> list[str]:
    return [key for key in (ADMIN_API_KEY,) if key]


def _key_matches(candidate: str, keys: list[str]) -> bool:
    return any(hmac.compare_digest(candidate, key) for key in keys)


def _required_keys_for_path(path: str) -> tuple[str, list[str]]:
    if path in ADMIN_PATHS or path.startswith("/admin/"):
        return "admin", _configured_admin_keys()
    return "read", _configured_read_keys()


def _is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    with _REQUEST_LOCK:
        queue = _REQUEST_LOG[client_ip]
        while queue and (now - queue[0]) > RATE_LIMIT_WINDOW_SEC:
            queue.popleft()
        if len(queue) >= RATE_LIMIT_REQUESTS:
            return True
        queue.append(now)
    return False


async def auth_and_rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    public_paths = {"/", "/status", "/docs", "/redoc", "/openapi.json"}
    if request.method == "OPTIONS" or path in public_paths or path.startswith("/web"):
        return await call_next(request)

    required_role, configured_keys = _required_keys_for_path(path)
    if required_role == "admin" and READ_API_KEY and not configured_keys:
        _audit_event("auth_failed", request, required_role=required_role, reason="admin_key_not_configured")
        return JSONResponse(status_code=403, content={"detail": "Chave administrativa nao configurada."})
    if configured_keys:
        request_key = request.headers.get("x-api-key", "")
        if not _key_matches(request_key, configured_keys):
            _audit_event("auth_failed", request, required_role=required_role)
            status_code = 403 if required_role == "admin" and request_key else 401
            return JSONResponse(status_code=status_code, content={"detail": "Chave de API invalida."})

    client_ip = (request.client.host if request.client else "") or "desconhecido"
    if _is_rate_limited(client_ip):
        _audit_event("rate_limited", request)
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    "Limite de requisicoes excedido. "
                    f"Tente novamente em alguns segundos (janela de {RATE_LIMIT_WINDOW_SEC}s)."
                )
            },
        )

    return await call_next(request)
