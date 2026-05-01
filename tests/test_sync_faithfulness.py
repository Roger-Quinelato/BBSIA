from __future__ import annotations

from bbsia.rag import pipeline


def _sample_results() -> list[dict]:
    return [
        {
            "score": 0.9,
            "score_dense": 0.8,
            "score_sparse": 1.0,
            "documento": "doc.pdf",
            "pagina": 1,
            "area": "ia",
            "assuntos": ["rag"],
            "texto": "O pipeline RAG recupera trechos e gera respostas com base neles.",
            "doc_autores": ["Maria Silva"],
            "doc_ano": 2026,
        }
    ]


def test_answer_question_skips_sync_faithfulness_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_SYNC_FAITHFULNESS", False)
    monkeypatch.setattr(pipeline, "search", lambda **_kwargs: _sample_results())
    monkeypatch.setattr(pipeline, "query_ollama", lambda **_kwargs: "Resposta util com base no contexto. (Silva, 2026)")
    monkeypatch.setattr(
        pipeline,
        "_faithfulness_check",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("faithfulness should be skipped")),
    )

    payload = pipeline.answer_question("Como funciona o pipeline RAG?")

    assert payload["resposta"] == "Resposta util com base no contexto. (Silva, 2026)"


def test_answer_question_runs_sync_faithfulness_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_SYNC_FAITHFULNESS", True)
    monkeypatch.setattr(pipeline, "search", lambda **_kwargs: _sample_results())
    monkeypatch.setattr(pipeline, "query_ollama", lambda **_kwargs: "Resposta nao suportada.")
    monkeypatch.setattr(pipeline, "_faithfulness_check", lambda *_args, **_kwargs: (False, "sem suporte"))

    payload = pipeline.answer_question("Como funciona o pipeline RAG?")

    assert "Com base nos trechos recuperados" in payload["resposta"]
    assert "Nao encontrei evidencias suficientes" not in payload["resposta"]


def test_answer_question_replaces_no_evidence_when_context_exists(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_SYNC_FAITHFULNESS", False)
    monkeypatch.setattr(pipeline, "search", lambda **_kwargs: _sample_results())
    monkeypatch.setattr(
        pipeline,
        "query_ollama",
        lambda **_kwargs: "Não encontrei evidências suficientes nos documentos indexados.",
    )

    payload = pipeline.answer_question("Como funciona o pipeline RAG?")

    assert "Com base nos trechos recuperados" in payload["resposta"]
    assert "Não encontrei evidências suficientes" not in payload["resposta"]
