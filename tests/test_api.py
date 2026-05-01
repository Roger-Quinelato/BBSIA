from __future__ import annotations

import json

from fastapi.testclient import TestClient

import api
from bbsia.app import core as api_core
import routers.rag as rag_router
import routers.system as system_router


def _sample_chunk() -> dict:
    return {
        "score": 0.91,
        "id": 88,
        "documento": "LIIA BBSIA - Infra-estrutura.pdf",
        "pagina": 3,
        "area": "infraestrutura",
        "assuntos": ["servidores", "kubernetes"],
        "doc_titulo": "Infraestrutura BBSIA",
        "doc_autores": ["Equipe Tecnica"],
        "doc_ano": 2026,
        "texto": "Trecho de teste",
        "chunk_index": 2,
    }


def _build_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        system_router,
        "cache_health",
        lambda load_if_empty=False: {
            "resources_cached": True,
            "embedding_model_loaded": True,
            "reranker_cached": False,
            "total_chunks": 2,
            "min_dense_score_percent": 18,
        },
    )
    monkeypatch.setattr(system_router, "list_available_areas", lambda: ["arquitetura", "infraestrutura"])
    monkeypatch.setattr(system_router, "list_available_assuntos", lambda: ["RAG", "kubernetes"])
    monkeypatch.setattr(system_router, "_check_ollama", lambda: (True, ["qwen3.5:7b-instruct"]))

    monkeypatch.setattr(api_core, "search", lambda query, top_k, filtro_area, filtro_assunto: [_sample_chunk()])
    monkeypatch.setattr(rag_router, "cache_health", lambda load_if_empty=False: {"resources_cached": True, "total_chunks": 2})

    async def _fake_stream(**kwargs):
        del kwargs
        yield {"type": "token", "token": "Resposta "}
        yield {"type": "token", "token": "de teste"}
        yield {"type": "metadata", "resultados": [_sample_chunk()], "fontes": ["fonte"]}

    monkeypatch.setattr(rag_router, "answer_question_stream", _fake_stream)
    monkeypatch.setattr(rag_router, "validate_ollama_model", lambda model: model)

    return TestClient(api.app)


def test_status_endpoint(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indice_carregado"] is True
    assert body["ollama_online"] is True


def test_rag_health_endpoint_expoe_cache(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.get("/rag/health")

    assert response.status_code == 200
    body = response.json()
    assert body["resources_cached"] is True
    assert body["total_chunks"] == 2


def test_search_endpoint(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.post(
            "/search",
            json={
                "query": "infraestrutura kubernetes",
                "top_k": 3,
                "filtro_area": ["infraestrutura"],
                "filtro_assunto": [],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["resultados"][0]["area"] == "infraestrutura"


def test_chat_stream_endpoint_returns_ndjson(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.post(
            "/chat",
            json={
                "pergunta": "Quais requisitos de infraestrutura?",
                "modelo": "qwen3.5:7b-instruct",
                "top_k": 3,
                "filtro_area": ["infraestrutura"],
                "filtro_assunto": [],
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert any(evt.get("type") == "token" for evt in events)
    assert any(evt.get("type") == "metadata" for evt in events)
