from __future__ import annotations

from pathlib import Path

from benchmarks.rag_benchmark import run_benchmark


def test_run_benchmark_computes_required_metrics(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id":"1","question":"q1","reference_answer":"a1","expected_context_terms":["kubernetes"]}',
                '{"id":"2","question":"q2","reference_answer":"a2","expected_context_terms":["lgpd"]}',
            ]
        ),
        encoding="utf-8",
    )

    def fake_answer_question(pergunta, top_k=5, filtro_area=None, filtro_assunto=None):
        del top_k, filtro_area, filtro_assunto
        if pergunta == "q1":
            return {
                "resposta": "A resposta cita kubernetes.",
                "fontes": ["doc1"],
                "resultados": [{"texto": "kubernetes e essencial"}],
            }
        return {
            "resposta": "A resposta cita lgpd.",
            "fontes": ["doc2"],
            "resultados": [{"texto": "lgpd exige base legal"}],
        }

    monkeypatch.setattr("benchmarks.rag_benchmark.rag_engine.answer_question", fake_answer_question)

    payload = run_benchmark(Path(dataset))

    assert payload["total_questions"] == 2
    assert "context_recall" in payload["metrics"]
    assert "faithfulness_score" in payload["metrics"]
    assert "answer_relevancy" in payload["metrics"]
    assert len(payload["details"]) == 2
