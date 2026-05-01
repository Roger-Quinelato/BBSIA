from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from bbsia.app.core import (
    ChatRequest,
    SearchRequest,
    SearchResponse,
    _CONVERSATION_HISTORY,
    _CONVERSATION_LOCK,
    _normalize_chunk,
    _raise_http_exception,
    answer_question_stream,
    cache_health,
    list_available_areas,
    list_available_assuntos,
    list_ollama_models,
    validate_ollama_model,
    DEFAULT_MODEL,
)

router = APIRouter(prefix="", tags=["RAG"])


@router.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    try:
        selected_model = validate_ollama_model(payload.modelo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    conversation_id = payload.conversation_id
    history = []
    if conversation_id:
        with _CONVERSATION_LOCK:
            history = _CONVERSATION_HISTORY.get(conversation_id, []).copy()

    async def generate():
        full_response = []
        try:
            async for chunk in answer_question_stream(
                pergunta=payload.pergunta,
                model=selected_model,
                top_k=payload.top_k,
                filtro_area=payload.filtro_area,
                filtro_assunto=payload.filtro_assunto,
                history=history,
            ):
                if chunk["type"] == "token":
                    full_response.append(chunk.get("token", ""))
                elif chunk["type"] == "metadata":
                    chunk["resultados"] = [_normalize_chunk(x) for x in chunk.get("resultados", [])]
                    chunk["modelo_usado"] = selected_model
                yield json.dumps(chunk) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        finally:
            if conversation_id and full_response:
                with _CONVERSATION_LOCK:
                    _CONVERSATION_HISTORY[conversation_id].append({"role": "user", "content": payload.pergunta})
                    _CONVERSATION_HISTORY[conversation_id].append(
                        {"role": "assistant", "content": "".join(full_response)}
                    )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/search", response_model=SearchResponse)
def semantic_search(payload: SearchRequest) -> SearchResponse:
    try:
        from bbsia.app.core import search

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
        raise


@router.get("/areas")
def get_areas() -> dict[str, list[str]]:
    try:
        return {"areas": list_available_areas()}
    except Exception as exc:
        _raise_http_exception(exc)
        raise


@router.get("/assuntos")
def get_assuntos() -> dict[str, list[str]]:
    try:
        return {"assuntos": list_available_assuntos()}
    except Exception as exc:
        _raise_http_exception(exc)
        raise


@router.get("/modelos")
def get_modelos() -> dict:
    try:
        modelos = list_ollama_models()
        return {"modelos": modelos, "default": DEFAULT_MODEL}
    except Exception as exc:
        _raise_http_exception(exc)
        raise


@router.get("/rag/health")
def get_rag_health(load: bool = False) -> dict:
    try:
        return cache_health(load_if_empty=load)
    except Exception as exc:
        return {
            "resources_cached": False,
            "embedding_model_loaded": False,
            "reranker_cached": False,
            "last_error": str(exc),
        }
