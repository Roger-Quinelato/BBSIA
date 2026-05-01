"""
Configuracoes por variavel de ambiente para o projeto BBSIA.

Carrega automaticamente um arquivo .env local (se existir) sem depender de
bibliotecas externas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Carrega variaveis de um arquivo .env usando setdefault."""
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, val = (x.strip() for x in line.split("=", 1))
        if key:
            if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}: val = val[1:-1]
            os.environ.setdefault(key, val)


def get_env_str(name: str, default: str) -> str:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else default


def get_env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [p.strip() for p in raw.split(",") if p.strip()] if raw is not None else default


def get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return raw.strip().lower() in {"1", "true", "yes", "on", "sim", "s"} if raw is not None else default


def get_env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        val = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    if min_value is not None: val = max(min_value, val)
    if max_value is not None: val = min(max_value, val)
    return val


@dataclass(frozen=True)
class ApiAuthUploadSettings:
    api_key: str
    read_api_key: str
    admin_api_key: str
    rate_limit_requests: int
    rate_limit_window_sec: int
    max_upload_size_mb: int
    max_pdf_pages: int
    max_pdf_extracted_chars: int
    pdf_validation_timeout_sec: int


@dataclass(frozen=True)
class OllamaGenerationSettings:
    ollama_url: str
    default_model: str
    allowed_llm_models: list[str]
    allow_remote_ollama: bool
    ollama_timeout_sec: int
    ollama_num_predict: int
    ollama_num_ctx: int


@dataclass(frozen=True)
class EmbeddingsSettings:
    embedding_model: str
    embedding_dim: int
    hf_local_files_only: bool


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int
    max_context_chunks: int
    enable_query_planning: bool
    hybrid_dense_candidates: int
    hybrid_sparse_candidates: int
    rrf_k: int
    min_dense_score_percent: int


@dataclass(frozen=True)
class RerankerSettings:
    enable_reranker: bool
    preload_reranker_on_startup: bool
    reranker_model: str
    reranker_candidates: int
    reranker_top_n: int
    reranker_max_length: int


@dataclass(frozen=True)
class IngestionSettings:
    preload_rag_on_startup: bool
    rag_health_load_on_status: bool


@dataclass(frozen=True)
class AppSettings:
    api_auth_upload: ApiAuthUploadSettings
    ollama_generation: OllamaGenerationSettings
    embeddings: EmbeddingsSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings
    ingestion: IngestionSettings


def load_settings() -> AppSettings:
    default_model = get_env_str("DEFAULT_MODEL", "qwen3.5:7b-instruct")
    api_key = get_env_str("API_KEY", "")
    return AppSettings(
        api_auth_upload=ApiAuthUploadSettings(
            api_key=api_key,
            read_api_key=get_env_str("READ_API_KEY", api_key),
            admin_api_key=get_env_str("ADMIN_API_KEY", api_key),
            rate_limit_requests=get_env_int("RATE_LIMIT_REQUESTS", 120, min_value=1, max_value=10000),
            rate_limit_window_sec=get_env_int("RATE_LIMIT_WINDOW_SEC", 60, min_value=1, max_value=3600),
            max_upload_size_mb=get_env_int("MAX_UPLOAD_SIZE_MB", 50, min_value=1, max_value=500),
            max_pdf_pages=get_env_int("MAX_PDF_PAGES", 300, min_value=1, max_value=5000),
            max_pdf_extracted_chars=get_env_int("MAX_PDF_EXTRACTED_CHARS", 2_000_000, min_value=1000, max_value=50_000_000),
            pdf_validation_timeout_sec=get_env_int("PDF_VALIDATION_TIMEOUT_SEC", 30, min_value=1, max_value=600),
        ),
        ollama_generation=OllamaGenerationSettings(
            ollama_url=get_env_str("OLLAMA_URL", "http://localhost:11434"),
            default_model=default_model,
            allowed_llm_models=get_env_list("ALLOWED_LLM_MODELS", [default_model]),
            allow_remote_ollama=get_env_bool("ALLOW_REMOTE_OLLAMA", False),
            ollama_timeout_sec=get_env_int("OLLAMA_TIMEOUT_SEC", 300, min_value=30, max_value=3600),
            ollama_num_predict=get_env_int("OLLAMA_NUM_PREDICT", 512, min_value=32, max_value=1024),
            ollama_num_ctx=get_env_int("OLLAMA_NUM_CTX", 8192, min_value=512, max_value=131072),
        ),
        embeddings=EmbeddingsSettings(
            embedding_model=get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
            embedding_dim=get_env_int("EMBEDDING_DIM", 1024, min_value=1, max_value=8192),
            hf_local_files_only=get_env_bool("HF_LOCAL_FILES_ONLY", True),
        ),
        retrieval=RetrievalSettings(
            top_k=get_env_int("TOP_K", 5, min_value=1, max_value=20),
            max_context_chunks=get_env_int("MAX_CONTEXT_CHUNKS", 6, min_value=1, max_value=10),
            enable_query_planning=get_env_bool("ENABLE_QUERY_PLANNING", False),
            hybrid_dense_candidates=get_env_int("HYBRID_DENSE_CANDIDATES", 40, min_value=5, max_value=2000),
            hybrid_sparse_candidates=get_env_int("HYBRID_SPARSE_CANDIDATES", 80, min_value=5, max_value=5000),
            rrf_k=get_env_int("RRF_K", 60, min_value=1, max_value=200),
            min_dense_score_percent=get_env_int("MIN_DENSE_SCORE_PERCENT", 18, min_value=0, max_value=100),
        ),
        reranker=RerankerSettings(
            enable_reranker=get_env_bool("ENABLE_RERANKER", True),
            preload_reranker_on_startup=get_env_bool("PRELOAD_RERANKER_ON_STARTUP", False),
            reranker_model=get_env_str("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
            reranker_candidates=get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200),
            reranker_top_n=get_env_int("RERANKER_TOP_N", 3, min_value=1, max_value=10),
            reranker_max_length=get_env_int("RERANKER_MAX_LENGTH", 512, min_value=128, max_value=2048),
        ),
        ingestion=IngestionSettings(
            preload_rag_on_startup=get_env_bool("PRELOAD_RAG_ON_STARTUP", True),
            rag_health_load_on_status=get_env_bool("RAG_HEALTH_LOAD_ON_STATUS", False),
        ),
    )


# Carrega .env no import antes de materializar settings.
load_env_file()
settings = load_settings()
