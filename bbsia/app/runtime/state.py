from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from pathlib import Path

from bbsia.core.config import get_env_bool, get_env_int, get_env_list, get_env_str
from bbsia.domain.document_metadata.service import (
    DEFAULT_UPLOAD_APPROVED_DIR,
    DEFAULT_UPLOAD_METADATA_FILE,
    DEFAULT_UPLOAD_QUARANTINE_DIR,
    DEFAULT_UPLOADS_DIR,
)

API_VERSION = "1.0.0"
DEFAULT_MODEL = get_env_str("DEFAULT_MODEL", "qwen3.5:7b-instruct")
OLLAMA_URL = get_env_str("OLLAMA_URL", "http://localhost:11434")
DEFAULT_TOP_K = get_env_int("TOP_K", 5, min_value=1, max_value=20)
CORS_ORIGINS = get_env_list(
    "CORS_ORIGINS",
    [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8501",
        "http://localhost:8502",
        "http://localhost:8000",
    ],
)
API_KEY = get_env_str("API_KEY", "")
READ_API_KEY = get_env_str("READ_API_KEY", API_KEY)
ADMIN_API_KEY = get_env_str("ADMIN_API_KEY", API_KEY)
RATE_LIMIT_REQUESTS = get_env_int("RATE_LIMIT_REQUESTS", 120, min_value=1, max_value=10000)
RATE_LIMIT_WINDOW_SEC = get_env_int("RATE_LIMIT_WINDOW_SEC", 60, min_value=1, max_value=3600)
MAX_UPLOAD_SIZE_MB = get_env_int("MAX_UPLOAD_SIZE_MB", 50, min_value=1, max_value=500)
MAX_PDF_PAGES = get_env_int("MAX_PDF_PAGES", 300, min_value=1, max_value=5000)
MAX_PDF_EXTRACTED_CHARS = get_env_int("MAX_PDF_EXTRACTED_CHARS", 2_000_000, min_value=1000, max_value=50_000_000)
PDF_VALIDATION_TIMEOUT_SEC = get_env_int("PDF_VALIDATION_TIMEOUT_SEC", 30, min_value=1, max_value=600)
RAG_HEALTH_LOAD_ON_STATUS = get_env_bool("RAG_HEALTH_LOAD_ON_STATUS", False)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"
UPLOADS_DIR = DEFAULT_UPLOADS_DIR
UPLOAD_QUARANTINE_DIR = DEFAULT_UPLOAD_QUARANTINE_DIR
UPLOAD_APPROVED_DIR = DEFAULT_UPLOAD_APPROVED_DIR
UPLOAD_METADATA_FILE = DEFAULT_UPLOAD_METADATA_FILE
AUDIT_LOG_FILE = DATA_DIR / "audit.log"

_REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
_REQUEST_LOCK = threading.Lock()
_AUDIT_LOCK = threading.Lock()
_CONVERSATION_HISTORY: dict[str, list[dict[str, str]]] = defaultdict(list)
_CONVERSATION_LOCK = threading.Lock()
LOGGER = logging.getLogger("bbsia.api")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

ADMIN_PATHS = {"/upload", "/upload-metadata", "/upload-legacy-disabled", "/reprocessar", "/recarregar"}
PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore as instrucoes anteriores",
    "system prompt",
    "developer message",
    "reveal your prompt",
    "revele seu prompt",
    "do not cite",
    "nao cite",
    "always answer",
    "responda sempre",
    "forget the context",
    "esqueca o contexto",
]
