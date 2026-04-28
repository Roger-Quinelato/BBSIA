from __future__ import annotations

import asyncio
import json

import pipeline
import respx
from httpx import Response


def _sample_results() -> list[dict]:
    return [
        {
            "score": 0.85,
            "score_dense": 0.85,
            "score_sparse": 0.1,
            "id": 1,
            "documento": "manual_infra.pdf",
            "pagina": 12,
            "area": "infraestrutura",
            "assuntos": ["kubernetes"],
            "doc_titulo": "Manual de Infra",
            "doc_autores": ["Equipe Plataforma"],
            "doc_ano": 2026,
            "texto": "Kubernetes exige monitoramento continuo e alta disponibilidade.",
            "chunk_index": 0,
        }
    ]


def test_answer_question_stream_emits_metadata_and_tokens(monkeypatch):
    assert pipeline.ENABLE_STREAM_FAITHFULNESS is False
    monkeypatch.setattr(pipeline, "ENABLE_STREAM_FAITHFULNESS", False)
    monkeypatch.setattr(pipeline, "search", lambda **kwargs: _sample_results())

    payload_lines = [
        json.dumps({"response": "Resposta ", "done": False}),
        json.dumps({"response": "final", "done": False}),
        json.dumps({"done": True}),
    ]

    async def _collect():
        events = []
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://localhost:11434/api/generate").mock(return_value=Response(200, text="\n".join(payload_lines)))
            async for event in pipeline.answer_question_stream(pergunta="Quais requisitos?", model="qwen3.5:7b-instruct"):
                events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[0]["type"] == "metadata"
    assert events[0]["resultados"][0]["documento"] == "manual_infra.pdf"
    token_text = "".join(e.get("token", "") for e in events if e.get("type") == "token")
    assert token_text == "Resposta final"
    assert not any(e.get("type") == "faithfulness" for e in events)


def test_answer_question_stream_emits_faithfulness_event_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_STREAM_FAITHFULNESS", True)
    monkeypatch.setattr(pipeline, "search", lambda **kwargs: _sample_results())
    monkeypatch.setattr(pipeline, "_faithfulness_check", lambda resposta, context: (False, "nao suportada"))

    payload_lines = [
        json.dumps({"response": "Resposta ", "done": False}),
        json.dumps({"response": "final", "done": False}),
        json.dumps({"done": True}),
    ]

    async def _collect():
        events = []
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://localhost:11434/api/generate").mock(return_value=Response(200, text="\n".join(payload_lines)))
            async for event in pipeline.answer_question_stream(pergunta="Quais requisitos?", model="qwen3.5:7b-instruct"):
                events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[0]["type"] == "metadata"
    assert any(event.get("type") == "token" for event in events)
    assert events[-1]["type"] == "faithfulness"
    assert events[-1]["faithfulness_checked"] is True
    assert events[-1]["faithful"] is False
    assert events[-1]["reason"] == "nao suportada"
    assert "fallback_response" in events[-1]


def test_answer_question_stream_emits_faithfulness_event_without_fallback_when_faithful(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_STREAM_FAITHFULNESS", True)
    monkeypatch.setattr(pipeline, "search", lambda **kwargs: _sample_results())
    monkeypatch.setattr(pipeline, "_faithfulness_check", lambda resposta, context: (True, None))

    payload_lines = [
        json.dumps({"response": "Resposta ", "done": False}),
        json.dumps({"response": "fiel", "done": False}),
        json.dumps({"done": True}),
    ]

    async def _collect():
        events = []
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://localhost:11434/api/generate").mock(return_value=Response(200, text="\n".join(payload_lines)))
            async for event in pipeline.answer_question_stream(pergunta="Quais requisitos?", model="qwen3.5:7b-instruct"):
                events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[-1]["type"] == "faithfulness"
    assert events[-1]["faithfulness_checked"] is True
    assert events[-1]["faithful"] is True
    assert events[-1]["reason"] is None
    assert "fallback_response" not in events[-1]


def test_answer_question_stream_returns_error_event_when_ollama_fails(monkeypatch):
    monkeypatch.setattr(pipeline, "search", lambda **kwargs: _sample_results())

    async def _collect():
        events = []
        with respx.mock(assert_all_called=True) as mock:
            mock.post("http://localhost:11434/api/generate").mock(return_value=Response(500, text="erro"))
            async for event in pipeline.answer_question_stream(pergunta="Quais requisitos?", model="qwen3.5:7b-instruct"):
                events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[0]["type"] == "metadata"
    assert any(event.get("type") == "error" for event in events)
