from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from bbsia.app.runtime.state import (
    API_VERSION,
    CORS_ORIGINS,
    LOGGER,
    UPLOAD_APPROVED_DIR,
    UPLOAD_METADATA_FILE,
    UPLOAD_QUARANTINE_DIR,
    UPLOADS_DIR,
    WEB_DIR,
)
from bbsia.app.security.auth import auth_and_rate_limit_middleware
from bbsia.domain.document_metadata.service import save_upload_metadata
from bbsia.rag.public_api.engine import PRELOAD_RAG_ON_STARTUP, PRELOAD_RERANKER_ON_STARTUP, preload_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(
    title="BBSIA RAG API",
    description="API REST para o Chatbot RAG do Banco Brasileiro de Solucoes de IA",
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

app.middleware("http")(auth_and_rate_limit_middleware)

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
