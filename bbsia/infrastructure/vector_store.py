from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np

COLLECTION_NAME = "bbsia_chunks"
COLLECTION_SOLUTIONS = "bbsia_solucoes"
QDRANT_LOCAL_DIRNAME = "qdrant_db"


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]


def _norm(value: str) -> str:
    return value.strip().lower()


def get_local_qdrant_client(data_dir: str):
    from qdrant_client import QdrantClient

    qdrant_path = os.path.join(data_dir, QDRANT_LOCAL_DIRNAME)
    return QdrantClient(path=qdrant_path)


def dense_ranked_candidates(
    query_vec: np.ndarray,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    qclient: Any,
    top_n: int,
    target_collection: str = COLLECTION_NAME,
) -> tuple[list[int], dict[int, float]]:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    areas = list({_norm(v) for v in _as_list(filtro_area) if v.strip()})
    assuntos = list({_norm(v) for v in _as_list(filtro_assunto) if v.strip()})

    must_conditions = []
    if areas:
        must_conditions.append(FieldCondition(key="area", match=MatchAny(any=areas)))
    if assuntos:
        must_conditions.append(FieldCondition(key="assuntos", match=MatchAny(any=assuntos)))

    query_filter = Filter(must=must_conditions) if must_conditions else None

    try:
        if hasattr(qclient, "search"):
            results = qclient.search(
                collection_name=target_collection,
                query_vector=query_vec.tolist(),
                query_filter=query_filter,
                limit=top_n,
            )
        else:
            response = qclient.query_points(
                collection_name=target_collection,
                query=query_vec.tolist(),
                query_filter=query_filter,
                limit=top_n,
            )
            results = response.points
    except Exception as exc:
        strict = os.getenv("RAG_STRICT_DENSE_ERRORS", "").strip().lower() in {"1", "true", "yes", "on", "sim", "s"}
        if strict:
            raise RuntimeError(f"Dense retrieval failed for Qdrant collection '{target_collection}': {exc}") from exc
        return [], {}

    ranked: list[int] = []
    score_map: dict[int, float] = {}
    for hit in results:
        doc_id = int(hit.id)
        ranked.append(doc_id)
        score_map[doc_id] = float(hit.score)

    return ranked, score_map


def vector_store_health(qclient: Any | None = None) -> dict[str, Any]:
    if qclient is None:
        return {"backend": "qdrant", "available": False, "collection": COLLECTION_NAME}

    try:
        exists = bool(qclient.collection_exists(COLLECTION_NAME))
        return {
            "backend": "qdrant",
            "available": True,
            "collection": COLLECTION_NAME,
            "collection_exists": exists,
        }
    except Exception as exc:
        return {
            "backend": "qdrant",
            "available": False,
            "collection": COLLECTION_NAME,
            "error": str(exc),
        }
