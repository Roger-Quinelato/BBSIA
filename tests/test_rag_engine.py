from __future__ import annotations

import requests

import rag_engine


def test_answer_question_uses_extractive_fallback_when_ollama_fails(monkeypatch):
    sample_results = [
        {
            "score": 0.9,
            "id": 1,
            "documento": "LIIA BBSIA - Infra-estrutura.pdf",
            "pagina": 3,
            "area": "infraestrutura",
            "assuntos": ["kubernetes"],
            "texto": "O projeto requer infraestrutura com foco em disponibilidade e seguranca.",
            "chunk_index": 0,
        }
    ]

    monkeypatch.setattr(rag_engine, "search", lambda *args, **kwargs: sample_results)
    monkeypatch.setattr(
        rag_engine,
        "query_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("timeout")),
    )

    result = rag_engine.answer_question(
        pergunta="Quais os requisitos de infraestrutura?",
        model="qwen3.5:7b-instruct",
        top_k=3,
    )

    assert "Nao foi possivel concluir a geracao" in result["resposta"]
    assert "LIIA BBSIA - Infra-estrutura.pdf" in result["resposta"]
    assert result["fontes"] == ["LIIA BBSIA - Infra-estrutura (p. 3)"]
    assert len(result["resultados"]) == 1


def test_faithfulness_check_requires_inline_citations():
    context = [
        {"texto": "Trecho A", "doc_autores": ["Maria Silva"], "doc_ano": 2024},
        {"texto": "Trecho B", "doc_autores": ["Joao Santos"], "doc_ano": 2025},
    ]

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho. (Silva, 2024)", context)
    assert ok is True
    assert reason is None

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho.", context)
    assert ok is False
    assert "citacoes" in reason

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho. (Outro, 2023)", context)
    assert ok is False
    assert "fora do contexto" in reason


def test_dedupe_by_parent_keeps_context_diverse():
    chunks = [
        {"parent_id": "p1"},
        {"parent_id": "p1"},
        {"parent_id": "p2"},
        {"parent_id": "p3"},
    ]

    assert rag_engine._dedupe_by_parent([0, 1, 2, 3], chunks, 3) == [0, 2, 3]


def test_cache_health_reports_cached_models(monkeypatch):
    fake_index = type("FakeIndex", (), {"d": 3, "ntotal": 2})()
    monkeypatch.setattr(
        rag_engine,
        "_CACHE",
        {
            "index": fake_index,
            "chunks": [{"id": 1}, {"id": 2}],
            "model": object(),
            "embeddings": None,
            "embedding_model": "fake-embedding",
            "reranker": object(),
            "loaded_at_utc": "2026-01-01T00:00:00+00:00",
        },
    )

    health = rag_engine.cache_health()

    assert health["resources_cached"] is True
    assert health["embedding_model_loaded"] is True
    assert health["reranker_cached"] is True
    assert health["embedding_model"] == "fake-embedding"
    assert health["total_chunks"] == 2
    assert health["embedding_dim"] == 3


def test_preload_resources_can_load_reranker(monkeypatch):
    data = {"chunks": [{"id": 1}], "model": object(), "embedding_model": "fake", "reranker": None}
    monkeypatch.setattr(rag_engine, "_load_resources", lambda: data)
    monkeypatch.setattr(rag_engine, "_get_reranker", lambda: object())
    monkeypatch.setattr(rag_engine, "ENABLE_RERANKER", True)

    result = rag_engine.preload_resources(load_reranker=True)

    assert result["status"] == "ok"
    assert result["embedding_model_loaded"] is True
    assert result["reranker_loaded"] is True
    assert result["total_chunks"] == 1


def test_calibrate_dense_threshold_uses_expected_queries():
    fixtures = {
        "infra": [
            {
                "documento": "infra.pdf",
                "area": "infraestrutura",
                "score": 0.71,
                "score_dense": 0.71,
                "score_sparse": 1.2,
            }
        ],
        "etica": [
            {
                "documento": "etica.pdf",
                "area": "juridico",
                "score": 0.64,
                "score_dense": 0.64,
                "score_sparse": 0.8,
            }
        ],
        "bolo": [
            {
                "documento": "infra.pdf",
                "area": "infraestrutura",
                "score": 0.05,
                "score_dense": 0.05,
                "score_sparse": 0.0,
            }
        ],
    }

    def fake_search(query, top_k=5):
        del top_k
        if "bolo" in query:
            return fixtures["bolo"]
        if "etica" in query:
            return fixtures["etica"]
        return fixtures["infra"]

    specs = [
        {"query": "infra", "area_esperada": "infraestrutura", "documento_esperado": "infra.pdf"},
        {"query": "etica", "area_esperada": "juridico", "documento_esperado": "etica.pdf"},
        {"query": "bolo", "area_esperada": "nenhuma"},
    ]

    payload = rag_engine.calibrate_dense_threshold(specs, top_k=3, search_fn=fake_search)

    assert payload["estatisticas"]["in_scope_count"] == 2
    assert payload["estatisticas"]["out_scope_count"] == 1
    assert payload["estatisticas"]["threshold_sugerido_percent"] >= 1
    assert payload["qualidade"]["passed"] == 3


def test_evaluate_retrieval_quality_flags_wrong_expected_area():
    def fake_search(query, top_k=5):
        del query, top_k
        return [{"documento": "infra.pdf", "area": "infraestrutura", "score_dense": 0.7, "score": 0.7}]

    payload = rag_engine.evaluate_retrieval_quality(
        [{"query": "LGPD", "area_esperada": "juridico"}],
        search_fn=fake_search,
    )

    assert payload["passed"] == 0
    assert payload["failed"] == 1
    assert payload["cases"][0]["area_ok"] is False
