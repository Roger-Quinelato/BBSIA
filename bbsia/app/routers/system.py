from typing import Any

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from bbsia.app.core import (
    API_VERSION,
    RAG_HEALTH_LOAD_ON_STATUS,
    WEB_DIR,
    _check_ollama,
    _reprocess_manager,
    cache_health,
    list_available_areas,
    list_available_assuntos,
)

router = APIRouter(prefix="", tags=["System"])


@router.get("/", response_model=None)
def root():
    if WEB_DIR.exists():
        return RedirectResponse(url="/web")
    return {"status": "ok", "mensagem": "API BBSIA ativa."}


@router.get("/status")
def status() -> dict[str, Any]:
    try:
        rag_cache = cache_health(load_if_empty=RAG_HEALTH_LOAD_ON_STATUS)
    except Exception as exc:
        rag_cache = {
            "resources_cached": False,
            "total_chunks": 0,
            "last_error": str(exc),
        }

    indice_carregado = bool(rag_cache.get("resources_cached"))
    total_chunks = int(rag_cache.get("total_chunks") or 0)
    total_areas = 0
    total_assuntos = 0
    if indice_carregado:
        try:
            total_areas = len(list_available_areas())
            total_assuntos = len(list_available_assuntos())
        except Exception:
            total_areas = 0
            total_assuntos = 0

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
        "rag_cache": rag_cache,
        "reprocessamento": _reprocess_manager.snapshot(),
    }
