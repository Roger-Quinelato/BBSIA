from __future__ import annotations

from bbsia.rag.retrieval from bbsia.rag.retrieval import retriever
from bbsia.infrastructure.vector_store import COLLECTION_NAME, COLLECTION_SOLUTIONS


def test_search_single_collection_uses_default(monkeypatch):
    called_collections = []

    def fake_search_single(query, top_k, filtro_area, filtro_assunto, collection):
        called_collections.append(collection)
        return [{"id": 1, "score": 0.9, "collection_used": collection}]

    monkeypatch.setattr(retriever, "_search_single_collection", fake_search_single)

    results = retriever.search("teste query")
    assert len(results) == 1
    assert called_collections == [COLLECTION_NAME]


def test_search_specific_collection(monkeypatch):
    called_collections = []

    def fake_search_single(query, top_k, filtro_area, filtro_assunto, collection):
        called_collections.append(collection)
        return [{"id": 2, "score": 0.8, "collection_used": collection}]

    monkeypatch.setattr(retriever, "_search_single_collection", fake_search_single)

    results = retriever.search("teste", target_collection=COLLECTION_SOLUTIONS)
    assert len(results) == 1
    assert called_collections == [COLLECTION_SOLUTIONS]


def test_search_multiple_collections_merges_and_sorts(monkeypatch):
    called_collections = []

    def fake_search_single(query, top_k, filtro_area, filtro_assunto, collection):
        called_collections.append(collection)
        if collection == COLLECTION_NAME:
            return [
                {"id": 1, "score": 0.5, "collection": collection},
                {"id": 2, "score": 0.9, "collection": collection},
            ]
        elif collection == COLLECTION_SOLUTIONS:
            return [
                {"id": 3, "score": 0.95, "collection": collection},
                {"id": 4, "score": 0.2, "collection": collection},
            ]
        return []

    monkeypatch.setattr(retriever, "_search_single_collection", fake_search_single)

    results = retriever.search("teste", top_k=3, target_collection=[COLLECTION_NAME, COLLECTION_SOLUTIONS])
    
    assert set(called_collections) == {COLLECTION_NAME, COLLECTION_SOLUTIONS}
    assert len(results) == 3
    
    # Must be sorted by score descending
    assert results[0]["id"] == 3  # 0.95
    assert results[1]["id"] == 2  # 0.9
    assert results[2]["id"] == 1  # 0.5
