from __future__ import annotations

from bbsia.rag.generation import generator
from bbsia.rag import pipeline


def _solution_result() -> dict:
    return {
        "score": 0.9,
        "score_dense": 0.9,
        "score_sparse": 0.1,
        "id": 1,
        "documento": "bbsia/domain/catalogo/data/solucoes_piloto.json#solucao-x",
        "pagina": None,
        "area": "tecnologia",
        "assuntos": ["piloto"],
        "doc_titulo": "Solucao X",
        "doc_autores": ["BBSIA"],
        "doc_ano": None,
        "texto": "Nome: Solucao X\nCausa raiz: triagem manual.\nPassos de implantacao: configurar API.\nRiscos: baixa precisao.",
        "chunk_index": 0,
    }


def _document_result() -> dict:
    return {
        "score": 0.8,
        "score_dense": 0.8,
        "score_sparse": 0.1,
        "id": 2,
        "documento": "manual.pdf",
        "pagina": 3,
        "area": "tecnologia",
        "assuntos": ["apoio"],
        "doc_titulo": "Manual",
        "doc_autores": ["Equipe"],
        "doc_ano": 2026,
        "texto": "Evidencia documental de apoio.",
        "chunk_index": 0,
    }


def test_build_prompt_adds_diagnostic_contract():
    context = (
        "--- SOLUCOES CANDIDATAS ---\n"
        "Fonte 1: Solucao X\n\n"
        "--- EVIDENCIAS DOCUMENTAIS ---\n"
        "Fonte 2: Manual"
    )

    prompt = generator.build_prompt("Como resolver erro na triagem?", context)

    assert "Modo diagnostico de problemas" in prompt
    assert "Diagnostico, Solucao Recomendada, Passos, Riscos" in prompt
    assert "Nao invente causa raiz, passos, riscos" in prompt
    assert generator.NO_SOLUTION_RESPONSE in prompt


def test_diagnostic_answer_falls_back_when_catalog_has_no_solution(monkeypatch):
    def fake_search(**kwargs):
        if kwargs.get("target_collection") == pipeline.COLLECTION_SOLUTIONS:
            return []
        return [_document_result()]

    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(
        pipeline,
        "query_ollama",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called without catalog solution")),
    )

    payload = pipeline.answer_question("Tenho erro e demora na triagem, qual solucao recomenda?")

    assert payload["diagnostic_mode"] is True
    assert payload["resposta"] == generator.NO_SOLUTION_RESPONSE
    assert payload["prompt"] is None
    assert payload["resultados"][0]["retrieval_domain"] == "documentos"


def test_diagnostic_answer_uses_structured_prompt_when_solution_exists(monkeypatch):
    def fake_search(**kwargs):
        if kwargs.get("target_collection") == pipeline.COLLECTION_SOLUTIONS:
            return [_solution_result()]
        return [_document_result()]

    monkeypatch.setattr(pipeline, "search", fake_search)
    captured = {}

    def fake_query_ollama(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Diagnostico\nSolucao Recomendada\nPassos\nRiscos"

    monkeypatch.setattr(pipeline, "query_ollama", fake_query_ollama)

    payload = pipeline.answer_question("Tenho erro e demora na triagem, qual solucao recomenda?")

    assert payload["diagnostic_mode"] is True
    assert "--- SOLUCOES CANDIDATAS ---" in captured["prompt"]
    assert "--- EVIDENCIAS DOCUMENTAIS ---" in captured["prompt"]
    assert "Diagnostico, Solucao Recomendada, Passos, Riscos" in captured["prompt"]
