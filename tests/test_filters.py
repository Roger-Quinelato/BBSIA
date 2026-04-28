from __future__ import annotations

import numpy as np

import retriever
from query_planning import plan_query
from rag_engine import _filter_ids


def test_filter_ids_by_area_and_assunto():
    chunks = [
        {"area": "infraestrutura", "assuntos": ["kubernetes", "seguranca"]},
        {"area": "etica", "assuntos": ["lgpd", "conformidade"]},
        {"area": "arquitetura", "assuntos": ["rag", "chatbot"]},
    ]

    assert _filter_ids(chunks, filtro_area=["infraestrutura"]) == [0]
    assert _filter_ids(chunks, filtro_assunto=["LGPD"]) == [1]
    assert _filter_ids(chunks, filtro_area=["arquitetura"], filtro_assunto=["chatbot"]) == [2]
    assert _filter_ids(chunks, filtro_area=["metodologia"]) == []


def test_plan_query_infers_area():
    plan = plan_query("Quais requisitos de infraestrutura para Kubernetes?")

    assert plan.filtro_area == "infraestrutura"
    assert plan.confidence > 0.0


def test_plan_query_infers_assunto():
    plan = plan_query("Como a LGPD trata dados pessoais?")

    assert plan.filtro_assunto == "lgpd"


def test_search_query_planning_does_not_override_explicit_filter(monkeypatch):
    captured: dict[str, object] = {}

    class FakeModel:
        def encode(self, *_args, **_kwargs):
            return np.array([[1.0]], dtype=np.float32)

    monkeypatch.setattr(retriever, "ENABLE_QUERY_PLANNING", True)
    monkeypatch.setattr(retriever, "EXPECTED_EMBEDDING_DIM", 1)
    monkeypatch.setattr(
        retriever,
        "_load_resources",
        lambda: {
            "chunks": [
                {
                    "id": 0,
                    "area": "juridico",
                    "assuntos": ["lgpd", "conformidade"],
                    "texto": "LGPD e privacidade.",
                }
            ],
            "model": FakeModel(),
            "qclient": object(),
            "token_counts": [retriever.Counter({"lgpd": 1})],
            "doc_lengths": [1],
            "doc_freq": {"lgpd": 1},
            "avgdl": 1.0,
        },
    )

    def fake_dense_ranked_candidates(query_vec, filtro_area, filtro_assunto, qclient, top_n):
        del query_vec, qclient, top_n
        captured["filtro_area"] = filtro_area
        captured["filtro_assunto"] = filtro_assunto
        return [0], {0: 0.9}

    monkeypatch.setattr(retriever, "_dense_ranked_candidates", fake_dense_ranked_candidates)
    monkeypatch.setattr(retriever, "_rerank_candidates", lambda query, candidate_ids, chunks: (candidate_ids, {}))

    results = retriever.search(
        "LGPD em infraestrutura",
        filtro_area="juridico",
        filtro_assunto="conformidade",
    )

    assert len(results) == 1
    assert captured["filtro_area"] == "juridico"
    assert captured["filtro_assunto"] == "conformidade"


def test_search_query_planning_disabled_by_default(monkeypatch):
    monkeypatch.setattr(retriever, "ENABLE_QUERY_PLANNING", False)
    monkeypatch.setattr(retriever, "plan_query", lambda _query: (_ for _ in ()).throw(AssertionError("unused")))
    monkeypatch.setattr(
        retriever,
        "_load_resources",
        lambda: (_ for _ in ()).throw(RuntimeError("stops after planning gate")),
    )

    raised = False
    try:
        retriever.search("LGPD")
    except RuntimeError as exc:
        raised = True
        assert str(exc) == "stops after planning gate"
    assert raised is True
