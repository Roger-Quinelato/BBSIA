from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bbsia.rag.public_api import engine as rag_engine


TOKEN_RE = re.compile(r"\b[\w\-]{2,}\b", re.IGNORECASE)


@dataclass
class EvalRow:
    id: str
    question: str
    reference_answer: str
    expected_context_terms: list[str]
    expected_solution_ids: list[str]
    filtro_area: list[str]
    filtro_assunto: list[str]


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "")}


def _load_dataset(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return rows

    if raw_text.startswith("["):
        raw_rows = json.loads(raw_text)
        if not isinstance(raw_rows, list):
            raise ValueError("Dataset JSON deve ser uma lista de casos.")
    else:
        raw_rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    for raw in raw_rows:
        expected_solution_ids = [
            str(x).strip()
            for x in raw.get("expected_solution_ids", [])
            if str(x).strip()
        ]
        if raw.get("expected_solution_id"):
            expected_solution_ids.append(str(raw["expected_solution_id"]).strip())
        rows.append(
            EvalRow(
                id=str(raw.get("id") or f"row-{len(rows)+1}"),
                question=str(raw["question"]),
                reference_answer=str(raw.get("reference_answer", "")),
                expected_context_terms=[str(x).strip().lower() for x in raw.get("expected_context_terms", []) if str(x).strip()],
                expected_solution_ids=expected_solution_ids,
                filtro_area=[str(x) for x in raw.get("filtro_area", [])],
                filtro_assunto=[str(x) for x in raw.get("filtro_assunto", [])],
            )
        )
    return rows


def _context_recall(expected_terms: list[str], retrieved_contexts: list[str]) -> float:
    if not expected_terms:
        return 1.0
    merged_context = "\n".join(retrieved_contexts).lower()
    hits = sum(1 for term in expected_terms if term in merged_context)
    return hits / len(expected_terms)


def _faithfulness_heuristic(answer: str, retrieved_contexts: list[str]) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _tokens("\n".join(retrieved_contexts))
    if not context_tokens:
        return 0.0
    overlap = len(answer_tokens.intersection(context_tokens))
    return overlap / len(answer_tokens)


def _answer_relevancy_heuristic(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens.intersection(a_tokens)) / len(q_tokens)


def _solution_match(expected_solution_ids: list[str], results: list[dict[str, Any]]) -> float:
    if not expected_solution_ids:
        return 1.0

    expected = {solution_id.strip() for solution_id in expected_solution_ids if solution_id.strip()}
    returned: set[str] = set()
    for item in results:
        solution_id = str(item.get("solution_id") or "").strip()
        if solution_id:
            returned.add(solution_id)
        documento = str(item.get("documento") or "")
        for expected_id in expected:
            if expected_id and expected_id in documento:
                returned.add(expected_id)

    return 1.0 if expected.intersection(returned) else 0.0


def _try_deepeval_scores(question: str, answer: str, contexts: list[str]) -> tuple[float | None, float | None]:
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except Exception:
        return None, None

    try:
        tc = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
        faithfulness_metric = FaithfulnessMetric(threshold=0.0)
        answer_rel_metric = AnswerRelevancyMetric(threshold=0.0)
        faithfulness_metric.measure(tc)
        answer_rel_metric.measure(tc)
        return float(faithfulness_metric.score), float(answer_rel_metric.score)
    except Exception:
        return None, None


def run_benchmark(dataset_path: Path) -> dict[str, Any]:
    rows = _load_dataset(dataset_path)
    details: list[dict[str, Any]] = []

    context_recalls: list[float] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    solution_match_scores: list[float] = []

    for row in rows:
        answer_payload = rag_engine.answer_question(
            pergunta=row.question,
            top_k=5,
            filtro_area=row.filtro_area,
            filtro_assunto=row.filtro_assunto,
        )

        results = answer_payload.get("resultados", [])
        contexts = [str(item.get("parent_text") or item.get("texto") or "") for item in results]
        answer = str(answer_payload.get("resposta", ""))

        context_recall = _context_recall(row.expected_context_terms, contexts)

        faithfulness_llm, answer_rel_llm = _try_deepeval_scores(row.question, answer, contexts)
        faithfulness = faithfulness_llm if faithfulness_llm is not None else _faithfulness_heuristic(answer, contexts)
        answer_rel = answer_rel_llm if answer_rel_llm is not None else _answer_relevancy_heuristic(row.question, answer)
        solution_match = _solution_match(row.expected_solution_ids, results)

        context_recalls.append(context_recall)
        faithfulness_scores.append(faithfulness)
        answer_relevancy_scores.append(answer_rel)
        solution_match_scores.append(solution_match)

        details.append(
            {
                "id": row.id,
                "question": row.question,
                "context_recall": round(context_recall, 4),
                "faithfulness_score": round(faithfulness, 4),
                "answer_relevancy": round(answer_rel, 4),
                "solution_match": round(solution_match, 4),
                "expected_solution_ids": row.expected_solution_ids,
                "retrieved_sources": answer_payload.get("fontes", []),
            }
        )

    total = len(rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "total_questions": total,
        "metrics": {
            "context_recall": round(sum(context_recalls) / total, 4) if total else 0.0,
            "faithfulness_score": round(sum(faithfulness_scores) / total, 4) if total else 0.0,
            "answer_relevancy": round(sum(answer_relevancy_scores) / total, 4) if total else 0.0,
            "solution_match": round(sum(solution_match_scores) / total, 4) if total else 0.0,
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de qualidade RAG (context recall, faithfulness, answer relevancy)")
    parser.add_argument(
        "--dataset",
        default="bbsia/evaluation/benchmarks/datasets/rag_eval_dataset.jsonl",
        help="Caminho do dataset JSONL",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Arquivo de saida JSON. Se vazio, usa bbsia/evaluation/benchmarks/results/rag_benchmark_<timestamp>.json",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

    payload = run_benchmark(dataset_path)

    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("bbsia/evaluation/benchmarks/results") / f"rag_benchmark_{timestamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["metrics"], ensure_ascii=False))
    print(f"Resultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
