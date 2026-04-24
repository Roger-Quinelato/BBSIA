from __future__ import annotations

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
