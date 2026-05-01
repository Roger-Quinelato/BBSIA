from __future__ import annotations

import requests

from bbsia.rag.generation import faithfulness
from bbsia.rag import pipeline
from bbsia.rag.retrieval import retriever


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

    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: sample_results)
    monkeypatch.setattr(
        pipeline,
        "query_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("timeout")),
    )

    result = pipeline.answer_question(
        pergunta="Quais os requisitos de infraestrutura?",
        model="qwen3.5:7b-instruct",
        top_k=3,
    )

    assert "Com base nos trechos recuperados" in result["resposta"]
    assert "LIIA BBSIA - Infra-estrutura.pdf" in result["resposta"]
    assert result["fontes"] == ["LIIA BBSIA - Infra-estrutura (p. 3)"]
    assert len(result["resultados"]) == 1


def test_faithfulness_check_flags_unsupported_sentence(monkeypatch):
    context = [
        {"texto": "Trecho A", "doc_autores": ["Maria Silva"], "doc_ano": 2024},
        {"texto": "Trecho B", "doc_autores": ["Joao Santos"], "doc_ano": 2025},
    ]

    class _FakeNLI:
        def predict(self, pairs):
            logits = []
            for _premise, hypothesis in pairs:
                hyp = hypothesis.lower()
                if "nao suportada" in hyp:
                    logits.append([0.1, 0.0, 0.9])
                elif "suportada" in hyp:
                    logits.append([0.1, 0.9, 0.0])
                else:
                    logits.append([0.1, 0.0, 0.9])
            return logits

    monkeypatch.setattr(faithfulness, "_get_nli_model", lambda: _FakeNLI())

    ok, reason = faithfulness._faithfulness_check("A afirmacao suportada esta no trecho.", context)
    assert ok is True
    assert reason is None

    ok, reason = faithfulness._faithfulness_check("A afirmacao nao suportada extrapola o contexto.", context)
    assert ok is False
    assert "nao suportada" in reason


def test_dedupe_by_parent_keeps_context_diverse():
    chunks = [
        {"parent_id": "p1"},
        {"parent_id": "p1"},
        {"parent_id": "p2"},
        {"parent_id": "p3"},
    ]

    assert retriever._dedupe_by_parent([0, 1, 2, 3], chunks, 3) == [0, 2, 3]


def test_attach_parent_text_recomposes_metadata_chunks():
    chunks = [
        {"id": 0, "parent_id": "parent-0", "texto": "filho"},
        {"id": 1, "parent_id": "parent-1", "texto": "filho com parent", "parent_text": "ja existe"},
    ]

    retriever._attach_parent_text(chunks, {"parent-0": "parent completo", "parent-1": "novo parent"})

    assert chunks[0]["parent_text"] == "parent completo"
    assert chunks[1]["parent_text"] == "ja existe"


def test_cache_health_reports_cached_models(monkeypatch):
    fake_data = {
        "chunks": [{"id": 1}, {"id": 2}],
        "model": object(),
        "embeddings": None,
        "embedding_model": "fake-embedding",
        "reranker": object(),
        "loaded_at_utc": "2026-01-01T00:00:00+00:00",
    }

    monkeypatch.setattr(retriever.index_store, "has_data", lambda: True)
    monkeypatch.setattr(retriever.index_store, "get_data_if_loaded", lambda: fake_data)
    monkeypatch.setattr(retriever.index_store, "get_status", lambda key: None)

    health = retriever.cache_health()

    assert health["resources_cached"] is True
    assert health["embedding_model_loaded"] is True
    assert health["reranker_cached"] is True
    assert health["embedding_model"] == "fake-embedding"
    assert health["total_chunks"] == 2


def test_preload_resources_can_load_reranker(monkeypatch):
    data = {"chunks": [{"id": 1}], "model": object(), "embedding_model": "fake", "reranker": None}
    monkeypatch.setattr(retriever, "_load_resources", lambda: data)
    monkeypatch.setattr(retriever, "_get_reranker", lambda: object())
    monkeypatch.setattr(retriever, "ENABLE_RERANKER", True)

    result = retriever.preload_resources(load_reranker=True)

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

    payload = retriever.calibrate_dense_threshold(specs, top_k=3, search_fn=fake_search)

    assert payload["estatisticas"]["in_scope_count"] == 2
    assert payload["estatisticas"]["out_scope_count"] == 1
    assert payload["estatisticas"]["threshold_sugerido_percent"] >= 1
    assert payload["qualidade"]["passed"] == 3


def test_evaluate_retrieval_quality_flags_wrong_expected_area():
    def fake_search(query, top_k=5):
        del query, top_k
        return [{"documento": "infra.pdf", "area": "infraestrutura", "score_dense": 0.7, "score": 0.7}]

    payload = retriever.evaluate_retrieval_quality(
        [{"query": "LGPD", "area_esperada": "juridico"}],
        search_fn=fake_search,
    )

    assert payload["passed"] == 0
    assert payload["failed"] == 1
    assert payload["cases"][0]["area_ok"] is False
