from __future__ import annotations

import numpy as np
import pytest

from bbsia.infrastructure.vector_store import dense_ranked_candidates


class _FailingQdrant:
    def search(self, **_kwargs):
        raise RuntimeError("collection missing")


def test_dense_ranked_candidates_keeps_compatibility_when_not_strict(monkeypatch):
    monkeypatch.delenv("RAG_STRICT_DENSE_ERRORS", raising=False)

    ranked, scores = dense_ranked_candidates(
        query_vec=np.array([1.0], dtype=np.float32),
        filtro_area=None,
        filtro_assunto=None,
        qclient=_FailingQdrant(),
        top_n=5,
    )

    assert ranked == []
    assert scores == {}


def test_dense_ranked_candidates_raises_in_strict_diagnostic_mode(monkeypatch):
    monkeypatch.setenv("RAG_STRICT_DENSE_ERRORS", "true")

    with pytest.raises(RuntimeError, match="Dense retrieval failed"):
        dense_ranked_candidates(
            query_vec=np.array([1.0], dtype=np.float32),
            filtro_area=None,
            filtro_assunto=None,
            qclient=_FailingQdrant(),
            top_n=5,
        )
