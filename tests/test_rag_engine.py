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
        model="llama3.1:8b",
        top_k=3,
    )

    assert "Nao foi possivel concluir a geracao" in result["resposta"]
    assert "LIIA BBSIA - Infra-estrutura.pdf" in result["resposta"]
    assert result["fontes"] == ["LIIA BBSIA - Infra-estrutura.pdf (p. 3)"]
    assert len(result["resultados"]) == 1


def test_faithfulness_check_requires_inline_citations():
    context = [{"texto": "Trecho A"}, {"texto": "Trecho B"}]

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho. [Fonte 1]", context)
    assert ok is True
    assert reason is None

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho.", context)
    assert ok is False
    assert "citacoes" in reason

    ok, reason = rag_engine._faithfulness_check("A resposta esta no trecho. [Fonte 3]", context)
    assert ok is False
    assert "inexistentes" in reason


def test_dedupe_by_parent_keeps_context_diverse():
    chunks = [
        {"parent_id": "p1"},
        {"parent_id": "p1"},
        {"parent_id": "p2"},
        {"parent_id": "p3"},
    ]

    assert rag_engine._dedupe_by_parent([0, 1, 2, 3], chunks, 3) == [0, 2, 3]
