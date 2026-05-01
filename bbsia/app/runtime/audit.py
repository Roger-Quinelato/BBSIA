from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from bbsia.app.runtime.state import AUDIT_LOG_FILE, DATA_DIR, LOGGER, _AUDIT_LOCK
from bbsia.core.observability import log_event


def _client_ip(request: Request | None) -> str:
    if request is None or request.client is None:
        return "desconhecido"
    return request.client.host or "desconhecido"


def _audit_event(event: str, request: Request | None = None, **details: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "client_ip": _client_ip(request),
    }
    if request is not None:
        payload["method"] = request.method
        payload["path"] = request.url.path
    payload.update(details)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
        with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _record_event(
    event: str,
    request: Request | None = None,
    level: int = logging.INFO,
    **details: Any,
) -> None:
    _audit_event(event, request, **details)
    log_event(LOGGER, "app.audit", event, level=level, **details)
