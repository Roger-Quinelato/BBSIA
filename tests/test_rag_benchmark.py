from __future__ import annotations

import json
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
    assert "solution_match" in payload["metrics"]
    assert len(payload["details"]) == 2


def test_run_benchmark_accepts_json_dataset_and_scores_solution_match(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "problema-1",
                    "question": "q1",
                    "reference_answer": "a1",
                    "expected_solution_id": "solucao-x",
                    "expected_context_terms": ["triagem"],
                },
                {
                    "id": "problema-2",
                    "question": "q2",
                    "reference_answer": "a2",
                    "expected_solution_id": "solucao-y",
                    "expected_context_terms": ["contratos"],
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_answer_question(pergunta, top_k=5, filtro_area=None, filtro_assunto=None):
        del top_k, filtro_area, filtro_assunto
        if pergunta == "q1":
            return {
                "resposta": "Diagnostico com triagem.",
                "fontes": ["catalogo"],
                "resultados": [{"solution_id": "solucao-x", "texto": "triagem"}],
            }
        return {
            "resposta": "Diagnostico com contratos.",
            "fontes": ["catalogo"],
            "resultados": [{"documento": "catalogo/solucoes_piloto.json#solucao-y", "texto": "contratos"}],
        }

    monkeypatch.setattr("benchmarks.rag_benchmark.rag_engine.answer_question", fake_answer_question)

    payload = run_benchmark(dataset)

    assert payload["total_questions"] == 2
    assert payload["metrics"]["solution_match"] == 1.0
    assert payload["details"][0]["expected_solution_ids"] == ["solucao-x"]
    assert payload["details"][1]["solution_match"] == 1.0


def test_eval_dataset_has_problem_queries_mapped_to_catalog_solutions():
    dataset_path = Path("benchmarks/eval_dataset.json")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_solution_ids = {
        "solucao-ia-classificacao-chamados",
        "solucao-ia-analise-contratos",
        "solucao-ia-triagem-saude",
    }

    assert len(rows) >= 5
    assert all(row.get("expected_solution_id") in expected_solution_ids for row in rows)
    assert all(row.get("reference_answer") for row in rows)
    assert all(row.get("expected_context_terms") for row in rows)
