from __future__ import annotations

from typing import Any

import requests
from fastapi import HTTPException

from bbsia.app.runtime.state import OLLAMA_URL


def _raise_http_exception(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail="Indice vetorial (Qdrant) nao encontrado. Execute /reprocessar primeiro.",
        ) from exc
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        raise HTTPException(
            status_code=502,
            detail=f"Ollama nao esta acessivel em {OLLAMA_URL}",
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
        from bbsia.rag.public_api.engine import validate_ollama_endpoint

        validate_ollama_endpoint()
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
        return True, sorted(models)
    except Exception:
        return False, []
