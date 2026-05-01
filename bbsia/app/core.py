"""
API REST do Chatbot RAG BBSIA.

Execucao:
  python -m bbsia.app.main
ou
  uvicorn bbsia.app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import hmac
import hashlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bbsia.core.config import get_env_bool, get_env_int, get_env_list, get_env_str
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from bbsia.rag.ingestion.chunking import run_chunking
from bbsia.rag.ingestion.embedding import run_embedding
from bbsia.rag.ingestion.extrator import run_extraction
from bbsia.rag.ingestion.worker import ReprocessWorker
from bbsia.rag.engine import (
    answer_question,
    answer_question_stream,
    cache_health,
    list_available_areas,
    list_available_assuntos,
    list_ollama_models,
    preload_resources,
    PRELOAD_RAG_ON_STARTUP,
    PRELOAD_RERANKER_ON_STARTUP,
    reload_resources,
    search,
    validate_ollama_model,
)
from bbsia.rag.ingestion.classificador import carregar_biblioteca


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
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOAD_QUARANTINE_DIR = UPLOADS_DIR / "quarantine"
UPLOAD_APPROVED_DIR = UPLOADS_DIR / "approved"
UPLOAD_METADATA_FILE = UPLOADS_DIR / "metadata_uploads.json"
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
        "ignore as instruções anteriores",
    "ignore as instrucoes anteriores",
    "system prompt",
    "developer message",
    "reveal your prompt",
    "revele seu prompt",
    "do not cite",
        "não cite",
    "nao cite",
    "always answer",
    "responda sempre",
    "forget the context",
        "esqueça o contexto",
    "esqueca o contexto",
]

PDF_VALIDATION_SCRIPT = r"""
import json
import sys
import fitz

path = sys.argv[1]
max_pages = int(sys.argv[2])
max_chars = int(sys.argv[3])
patterns = json.loads(sys.argv[4])

extracted_chars = 0
findings = set()

try:
    doc = fitz.open(path)
except Exception as exc:
    print(f"PDF invalido ou corrompido: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    page_count = int(doc.page_count)
    if page_count <= 0:
        print("PDF sem paginas validas.", file=sys.stderr)
        sys.exit(3)
    if page_count > max_pages:
        print(f"PDF excede o limite de {max_pages} paginas.", file=sys.stderr)
        sys.exit(4)

    for page in doc:
        text = page.get_text("text") or ""
        lower = text.lower()
        for pattern in patterns:
            if pattern in lower:
                findings.add(pattern)
        extracted_chars += len(text)
        if extracted_chars > max_chars:
            print(f"PDF excede o limite de {max_chars} caracteres extraidos.", file=sys.stderr)
            sys.exit(5)
finally:
    doc.close()

print(json.dumps({
    "page_count": page_count,
    "extracted_chars": extracted_chars,
    "prompt_injection_findings": sorted(findings),
}, ensure_ascii=False))
"""


@dataclass(frozen=True)
class PdfValidationResult:
    page_count: int
    extracted_chars: int
    prompt_injection_findings: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if PRELOAD_RAG_ON_STARTUP:
        try:
            preload_result = preload_resources(load_reranker=PRELOAD_RERANKER_ON_STARTUP)
            LOGGER.info("event=startup_rag_preloaded details=%s", json.dumps(preload_result, ensure_ascii=False))
        except Exception as exc:
            LOGGER.warning("event=startup_rag_preload_failed error=%s", exc)
    else:
        LOGGER.info("event=startup_rag_preload_skipped")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_QUARANTINE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_APPROVED_DIR, exist_ok=True)
    if not UPLOAD_METADATA_FILE.exists():
        save_upload_metadata({})

    yield
# Shutdown (nenhuma ação necessária por ora)


app = FastAPI(
    title="BBSIA RAG API",
    description="API REST para o Chatbot RAG do Banco Brasileiro de Soluções de IA",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    LOGGER.warning("prometheus-fastapi-instrumentator nao instalado. Metricas desativadas.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


class ChatRequest(BaseModel):
    pergunta: str
    modelo: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filtro_area: list[str] = Field(default_factory=list)
    filtro_assunto: list[str] = Field(default_factory=list)
    conversation_id: str | None = None

    @field_validator("pergunta")
    @classmethod
    def validar_pergunta(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("O campo 'pergunta' é obrigatório e não pode estar vazio.")
        return value.strip()


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filtro_area: list[str] = Field(default_factory=list)
    filtro_assunto: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def validar_query(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("O campo 'query' é obrigatório e não pode estar vazio.")
        return value.strip()


class ChunkResult(BaseModel):
    score: float
    id: int
    documento: str
    pagina: int | None
    area: str
    assuntos: list[str]
    doc_titulo: str = ""
    doc_autores: list[str] = Field(default_factory=list)
    doc_ano: int | None = None
    section_heading: str | None = None
    content_type: str = "text"
    parent_id: str | None = None
    ocr_usado: bool = False
    table_index: int | None = None
    texto: str
    chunk_index: int | None


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[str]
    resultados: list[ChunkResult]
    modelo_usado: str
    total_fontes: int
    total_chunks_recuperados: int


class SearchResponse(BaseModel):
    query: str
    total: int
    resultados: list[ChunkResult]


class UploadMetadataRequest(BaseModel):
    documento: str
    area: str = Field(min_length=1)
    assuntos: list[str] = Field(default_factory=list)

    @field_validator("documento", "area")
    @classmethod
    def validar_campos_texto(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Campo obrigatório não pode estar vazio.")
        return value.strip()

    @field_validator("assuntos")
    @classmethod
    def validar_assuntos(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return cleaned or ["geral"]


def normalize_upload_doc_name(doc_name: str) -> str:
    normalized = (doc_name or "").strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "uploads/desconhecido.pdf"
    if parts[0] == "uploads":
        return "/".join(parts)
    return f"uploads/{parts[-1]}"


def load_upload_metadata() -> dict[str, dict[str, Any]]:
    if not UPLOAD_METADATA_FILE.exists():
        return {}
    try:
        data = json.loads(UPLOAD_METADATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_upload_metadata(data: dict[str, dict[str, Any]]) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_METADATA_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_upload_metadata_entry(
    doc_name: str,
    area: str,
    assuntos: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = load_upload_metadata()
    key = normalize_upload_doc_name(doc_name)
    entry: dict[str, Any] = {
        "area": area.strip(),
        "assuntos": [a.strip() for a in assuntos if a.strip()] or ["geral"],
    }
    if extra:
        entry.update(extra)
    metadata[key] = entry
    save_upload_metadata(metadata)


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
    LOGGER.log(level, "event=%s details=%s", event, json.dumps(details, ensure_ascii=False, default=str))


def _run_reprocess_pipeline(mark_step: Any) -> None:
    stages: list[tuple[str, Any]] = [
        ("extracao", run_extraction),
        ("chunking", run_chunking),
        ("embedding", run_embedding),
        ("reload", reload_resources),
    ]

    for stage_name, stage_fn in stages:
        mark_step(stage_name)
        stage_started = time.perf_counter()
        _record_event("reprocess_stage_started", None, stage=stage_name)
        try:
            stage_result = stage_fn()
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - stage_started) * 1000)
            _record_event(
                "reprocess_stage_failed",
                None,
                level=logging.ERROR,
                stage=stage_name,
                duration_ms=elapsed_ms,
                error=str(exc),
            )
            raise

        elapsed_ms = int((time.perf_counter() - stage_started) * 1000)
        metrics: dict[str, Any] = stage_result if isinstance(stage_result, dict) else {}
        _record_event(
            "reprocess_stage_completed",
            None,
            stage=stage_name,
            duration_ms=elapsed_ms,
            metrics=metrics,
        )

        for item in metrics.get("documento_erros", []):
            if not isinstance(item, dict):
                continue
            _record_event(
                "reprocess_document_error",
                None,
                level=logging.ERROR,
                stage=stage_name,
                documento=item.get("documento", "desconhecido"),
                error=item.get("erro", "erro nao informado"),
            )


def _build_reprocess_manager() -> ReprocessWorker:
    def _on_worker_event(event: str, details: dict[str, Any]) -> None:
        _record_event(event, None, **details)

    return ReprocessWorker(pipeline=_run_reprocess_pipeline, on_event=_on_worker_event)


_reprocess_manager = _build_reprocess_manager()


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


def validate_pdf_upload(path: Path, timeout_sec: int = PDF_VALIDATION_TIMEOUT_SEC) -> PdfValidationResult:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                PDF_VALIDATION_SCRIPT,
                str(path),
                str(MAX_PDF_PAGES),
                str(MAX_PDF_EXTRACTED_CHARS),
                json.dumps(PROMPT_INJECTION_PATTERNS, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Validacao do PDF excedeu {timeout_sec}s.") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Validacao do PDF falhou.").strip()
        raise ValueError(detail)

    try:
        payload = json.loads(result.stdout)
        return PdfValidationResult(
            page_count=int(payload["page_count"]),
            extracted_chars=int(payload["extracted_chars"]),
            prompt_injection_findings=[str(item) for item in payload.get("prompt_injection_findings", [])],
        )
    except Exception as exc:
        raise ValueError("Validacao do PDF retornou saida invalida.") from exc


def _safe_quarantine_path(original_filename: str) -> Path:
    extension = Path(original_filename).suffix.lower()
    if extension != ".pdf":
        extension = ".pdf"
    filename = f"{secrets.token_hex(16)}{extension}"
    UPLOAD_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    destination = (UPLOAD_QUARANTINE_DIR / filename).resolve()
    quarantine_root = UPLOAD_QUARANTINE_DIR.resolve()
    if quarantine_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Destino de upload invalido.")
    return destination


def _safe_approved_path(stored_filename: str) -> Path:
    name = Path((stored_filename or "").strip()).name
    if not name or Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail="Nome de arquivo aprovado invalido.")
    UPLOAD_APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    destination = (UPLOAD_APPROVED_DIR / name).resolve()
    approved_root = UPLOAD_APPROVED_DIR.resolve()
    if approved_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Destino de aprovacao invalido.")
    return destination


def _metadata_key_for_stored_filename(filename: str, metadata: dict[str, Any]) -> str | None:
    for key, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stored_filename") == filename:
            return key
        if Path(str(key)).name == filename:
            return key
    return None


def _resolve_quarantine_source_path(quarantine_path: str, stored_filename: str) -> Path:
    candidates: list[Path] = []
    if quarantine_path:
        qpath = Path(quarantine_path)
        if qpath.is_absolute():
            candidates.append(qpath)
        else:
            candidates.append(BASE_DIR / qpath)
    candidates.append(UPLOAD_QUARANTINE_DIR / stored_filename)

    quarantine_root = UPLOAD_QUARANTINE_DIR.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if quarantine_root in resolved.parents and resolved.exists():
            return resolved

    # Retorna fallback dentro da quarentena mesmo quando o arquivo não existe
    return (UPLOAD_QUARANTINE_DIR / stored_filename).resolve()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@app.middleware("http")
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



def _raise_http_exception(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail="Indice vetorial (Qdrant) nao encontrado. Execute /reprocessar primeiro.",
        ) from exc
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        raise HTTPException(
            status_code=502,
            detail=f"Ollama não está acessível em {OLLAMA_URL}",
        ) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _normalize_chunk(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": float(item.get("score", 0.0)),
        "id": int(item.get("id", 0)),
        "documento": str(item.get("documento", "desconhecido")),
        "pagina": item.get("pagina"),
        "area": str(item.get("area", "geral")),
        "assuntos": [str(v) for v in item.get("assuntos", [])],
        "doc_titulo": str(item.get("doc_titulo", "")),
        "doc_autores": [str(v) for v in item.get("doc_autores", [])],
        "doc_ano": item.get("doc_ano"),
        "section_heading": item.get("section_heading"),
        "content_type": str(item.get("content_type", "text")),
        "parent_id": item.get("parent_id"),
        "ocr_usado": bool(item.get("ocr_usado", False)),
        "table_index": item.get("table_index"),
        "texto": str(item.get("texto", "")),
        "chunk_index": item.get("chunk_index"),
    }


def _check_ollama() -> tuple[bool, list[str]]:
    try:
        from bbsia.rag.engine import validate_ollama_endpoint  # noqa: PLC0415

        validate_ollama_endpoint()
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
        return True, sorted(models)
    except Exception:
        return False, []
