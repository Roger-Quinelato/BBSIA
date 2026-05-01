from __future__ import annotations

import json
import logging
from typing import Any


def log_event(
    logger: logging.Logger,
    module: str,
    event: str,
    level: int = logging.INFO,
    **details: Any,
) -> None:
    logger.log(
        level,
        "module=%s event=%s details=%s",
        module,
        event,
        json.dumps(details, ensure_ascii=False, default=str),
    )
