"""
API REST do Chatbot RAG BBSIA.

Execucao:
  python api.py
ou
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from config import get_env_int, get_env_list, get_env_str
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from chunking import run_chunking
from embedding import run_embedding
from extrator_pdf_v2 import run_extraction
from rag_engine import (
    answer_question,
    list_available_areas,
    list_available_assuntos,
    list_ollama_models,
    reload_resources,
    search,
)
from classificador_artigo import carregar_biblioteca


API_VERSION = "1.0.0"
DEFAULT_MODEL = get_env_str("DEFAULT_MODEL", "llama3.1:8b")
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
RATE_LIMIT_REQUESTS = get_env_int("RATE_LIMIT_REQUESTS", 120, min_value=1, max_value=10000)
RATE_LIMIT_WINDOW_SEC = get_env_int("RATE_LIMIT_WINDOW_SEC", 60, min_value=1, max_value=3600)
MAX_UPLOAD_SIZE_MB = get_env_int("MAX_UPLOAD_SIZE_MB", 50, min_value=1, max_value=500)
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOAD_METADATA_FILE = UPLOADS_DIR / "metadata_uploads.json"

_REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
_REQUEST_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        from rag_engine import _load_resources  # noqa: PLC0415

        _load_resources()
        print("Índice FAISS carregado com sucesso.")
    except Exception as exc:
        print(f"AVISO: Índice não carregado na inicialização: {exc}")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
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
    filename = normalized.split("/")[-1]
    return f"uploads/{filename}"


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


def update_upload_metadata_entry(doc_name: str, area: str, assuntos: list[str]) -> None:
    metadata = load_upload_metadata()
    key = normalize_upload_doc_name(doc_name)
    metadata[key] = {
        "area": area.strip(),
        "assuntos": [a.strip() for a in assuntos if a.strip()] or ["geral"],
    }
    save_upload_metadata(metadata)


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


@app.middleware("http")
async def auth_and_rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    public_paths = {"/", "/status", "/docs", "/redoc", "/openapi.json"}
    if request.method == "OPTIONS" or path in public_paths or path.startswith("/web"):
        return await call_next(request)

    if API_KEY:
        request_key = request.headers.get("x-api-key", "")
        if request_key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Chave de API inválida."})

    client_ip = (request.client.host if request.client else "") or "desconhecido"
    if _is_rate_limited(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    "Limite de requisições excedido. "
                    f"Tente novamente em alguns segundos (janela de {RATE_LIMIT_WINDOW_SEC}s)."
                )
            },
        )

    return await call_next(request)


@app.get("/")
def root():
    if WEB_DIR.exists():
        return RedirectResponse(url="/web")
    return {"status": "ok", "mensagem": "API BBSIA ativa."}


def _raise_http_exception(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail="Índice FAISS não encontrado. Execute /reprocessar primeiro.",
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
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
        return True, sorted(models)
    except Exception:
        return False, []



@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(
            pergunta=payload.pergunta,
            model=payload.modelo,
            top_k=payload.top_k,
            filtro_area=payload.filtro_area,
            filtro_assunto=payload.filtro_assunto,
        )
        resultados = [_normalize_chunk(x) for x in result.get("resultados", [])]
        fontes = [str(v) for v in result.get("fontes", [])]
        resposta = str(result.get("resposta", "")).strip()
        if not resposta:
            resposta = "Não foi possível gerar resposta no momento."

        return ChatResponse(
            resposta=resposta,
            fontes=fontes,
            resultados=resultados,
            modelo_usado=payload.modelo,
            total_fontes=len(fontes),
            total_chunks_recuperados=len(resultados),
        )
    except Exception as exc:
        # /chat precisa ser tolerante a falhas do LLM.
        try:
            resultados = search(
                query=payload.pergunta,
                top_k=payload.top_k,
                filtro_area=payload.filtro_area,
                filtro_assunto=payload.filtro_assunto,
            )
            chunks = [_normalize_chunk(x) for x in resultados]
            fontes = []
            seen = set()
            for item in chunks:
                src = f"{item['documento']} (p. {item['pagina']})"
                if src not in seen:
                    seen.add(src)
                    fontes.append(src)

            return ChatResponse(
                resposta=(
                    "Recuperação concluída, mas o modelo de linguagem não respondeu. "
                    f"Verifique o Ollama em {OLLAMA_URL} e tente novamente."
                ),
                fontes=fontes,
                resultados=chunks,
                modelo_usado=payload.modelo,
                total_fontes=len(fontes),
                total_chunks_recuperados=len(chunks),
            )
        except Exception as inner_exc:
            _raise_http_exception(inner_exc if not isinstance(exc, FileNotFoundError) else exc)
            raise  # pragma: no cover


@app.post("/search", response_model=SearchResponse)
def semantic_search(payload: SearchRequest) -> SearchResponse:
    try:
        resultados = search(
            query=payload.query,
            top_k=payload.top_k,
            filtro_area=payload.filtro_area,
            filtro_assunto=payload.filtro_assunto,
        )
        parsed = [_normalize_chunk(x) for x in resultados]
        return SearchResponse(query=payload.query, total=len(parsed), resultados=parsed)
    except Exception as exc:
        _raise_http_exception(exc)
        raise  # pragma: no cover


@app.get("/areas")
def get_areas() -> dict[str, list[str]]:
    try:
        return {"areas": list_available_areas()}
    except Exception as exc:
        _raise_http_exception(exc)
        raise  # pragma: no cover


@app.get("/assuntos")
def get_assuntos() -> dict[str, list[str]]:
    try:
        return {"assuntos": list_available_assuntos()}
    except Exception as exc:
        _raise_http_exception(exc)
        raise  # pragma: no cover


@app.get("/modelos")
def get_modelos() -> dict[str, Any]:
    try:
        modelos = list_ollama_models()
        return {"modelos": modelos, "default": DEFAULT_MODEL}
    except Exception as exc:
        _raise_http_exception(exc)
        raise  # pragma: no cover


_reprocessamento_status: dict[str, Any] = {
    "rodando": False,
    "ultima_etapa": None,
    "erro": None,
    "concluido_em": None,
}


def _run_reprocessamento() -> None:
    """Executa o pipeline completo de reprocessamento em background."""
    global _reprocessamento_status
    _reprocessamento_status = {
        "rodando": True,
        "ultima_etapa": None,
        "erro": None,
        "concluido_em": None,
    }
    try:
        run_extraction()
        _reprocessamento_status["ultima_etapa"] = "extracao"

        run_chunking()
        _reprocessamento_status["ultima_etapa"] = "chunking"

        run_embedding()
        _reprocessamento_status["ultima_etapa"] = "embedding"

        reload_resources()
        _reprocessamento_status["ultima_etapa"] = "reload"

        from datetime import datetime, timezone  # noqa: PLC0415

        _reprocessamento_status["concluido_em"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        _reprocessamento_status["erro"] = str(exc)
    finally:
        _reprocessamento_status["rodando"] = False


@app.post("/reprocessar")
def reprocessar_base(background_tasks: BackgroundTasks) -> dict[str, Any]:
    if _reprocessamento_status.get("rodando"):
        raise HTTPException(
            status_code=409,
            detail="Reprocessamento já em andamento. Consulte /status para acompanhar.",
        )
    background_tasks.add_task(_run_reprocessamento)
    return {
        "status": "iniciado",
        "mensagem": "Reprocessamento iniciado em background. Consulte /status para acompanhar.",
    }


@app.post("/recarregar")
def recarregar_indice() -> dict[str, str]:
    try:
        reload_resources()
        return {"status": "ok", "mensagem": "Indice FAISS recarregado em memoria."}
    except Exception as exc:
        _raise_http_exception(exc)
        raise  # pragma: no cover


@app.post("/upload")
def upload(
    files: list[UploadFile] = File(...),
    area: str | None = Form(default=None),
    assuntos: str | None = Form(default=None),
) -> dict[str, Any]:
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    salvos: list[str] = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=422, detail="Nome de arquivo inválido.")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail=f"Arquivo inválido: {file.filename}. Apenas .pdf é permitido.",
            )

        destino = UPLOADS_DIR / Path(file.filename).name
        content = file.file.read()
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Arquivo '{file.filename}' excede o limite de {MAX_UPLOAD_SIZE_MB} MB "
                    f"({len(content) / (1024 * 1024):.1f} MB recebidos)."
                ),
            )
        with open(destino, "wb") as out:
            out.write(content)
        salvos.append(destino.name)

    # Metadados opcionais aplicados a todos os arquivos do envio.
    if area:
        parsed_assuntos = []
        if assuntos:
            parsed_assuntos = [a.strip() for a in assuntos.split(",") if a.strip()]
        for filename in salvos:
            update_upload_metadata_entry(
                doc_name=filename,
                area=area,
                assuntos=parsed_assuntos or ["geral"],
            )

    return {
        "arquivos_salvos": salvos,
        "total": len(salvos),
        "mensagem": "Arquivos salvos em uploads/. Execute /reprocessar para incluir na base.",
    }


@app.post("/upload-metadata")
def upload_metadata(payload: UploadMetadataRequest) -> dict[str, Any]:
    update_upload_metadata_entry(
        doc_name=payload.documento,
        area=payload.area,
        assuntos=payload.assuntos,
    )
    return {
        "status": "ok",
        "mensagem": "Metadados de upload cadastrados com sucesso.",
        "documento": normalize_upload_doc_name(payload.documento),
        "area": payload.area,
        "assuntos": payload.assuntos,
    }


# ---------------------------------------------------------------------------
# Fase 3 — Endpoints de Biblioteca e Filtros
# ---------------------------------------------------------------------------


@app.get("/biblioteca")
def get_biblioteca(
    area: str | None = None,
    tipo: str | None = None,
    ano_min: int | None = None,
    ano_max: int | None = None,
) -> dict[str, Any]:
    """Retorna o catálogo de documentos com filtros opcionais."""
    biblioteca = carregar_biblioteca()
    docs = biblioteca.get("documentos", [])

    # Aplica filtros
    if area:
        area_lower = area.strip().lower()
        docs = [d for d in docs if str(d.get("area_tematica", "")).lower() == area_lower]
    if tipo:
        tipo_lower = tipo.strip().lower()
        docs = [d for d in docs if str(d.get("tipo_documento", "")).lower() == tipo_lower]
    if ano_min is not None:
        docs = [d for d in docs if isinstance(d.get("ano"), int) and d["ano"] >= ano_min]
    if ano_max is not None:
        docs = [d for d in docs if isinstance(d.get("ano"), int) and d["ano"] <= ano_max]

    # Retorna campos resumidos (sem resumo e seções para listagem)
    resumidos = []
    for d in docs:
        resumidos.append({
            "id": d.get("id", ""),
            "titulo": d.get("titulo", ""),
            "autores": d.get("autores", []),
            "ano": d.get("ano"),
            "area_tematica": d.get("area_tematica", "geral"),
            "assuntos": d.get("assuntos", []),
            "tipo_documento": d.get("tipo_documento", "outro"),
            "paginas_total": d.get("paginas_total", 0),
        })

    return {"total": len(resumidos), "documentos": resumidos}


@app.get("/biblioteca/{doc_id}")
def get_biblioteca_doc(doc_id: str) -> dict[str, Any]:
    """Retorna metadado completo de um documento específico."""
    biblioteca = carregar_biblioteca()
    for doc in biblioteca.get("documentos", []):
        if doc.get("id") == doc_id:
            return doc
    raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado na biblioteca.")


@app.get("/filtros")
def get_filtros() -> dict[str, Any]:
    """Retorna valores únicos disponíveis para popular filtros dinâmicos no frontend."""
    biblioteca = carregar_biblioteca()
    docs = biblioteca.get("documentos", [])

    areas: set[str] = set()
    tipos: set[str] = set()
    anos: set[int] = set()
    assuntos: set[str] = set()

    for d in docs:
        area = d.get("area_tematica", "")
        if area:
            areas.add(area)
        tipo = d.get("tipo_documento", "")
        if tipo:
            tipos.add(tipo)
        ano = d.get("ano")
        if isinstance(ano, int):
            anos.add(ano)
        for a in d.get("assuntos", []):
            if isinstance(a, str) and a.strip():
                assuntos.add(a.strip())

    return {
        "areas": sorted(areas),
        "tipos": sorted(tipos),
        "anos": sorted(anos),
        "assuntos": sorted(assuntos),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    indice_carregado = False
    total_chunks = 0
    total_areas = 0
    total_assuntos = 0

    try:
        from rag_engine import _load_resources  # noqa: PLC0415

        data = _load_resources()
        indice_carregado = True
        total_chunks = len(data.get("chunks", []))
        total_areas = len(list_available_areas())
        total_assuntos = len(list_available_assuntos())
    except Exception:
        indice_carregado = False

    ollama_online, modelos = _check_ollama()

    return {
        "status": "ok",
        "indice_carregado": indice_carregado,
        "total_chunks": total_chunks,
        "areas_disponiveis": total_areas,
        "assuntos_disponiveis": total_assuntos,
        "ollama_online": ollama_online,
        "modelos_disponiveis": modelos,
        "versao_api": API_VERSION,
        "reprocessamento": _reprocessamento_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
