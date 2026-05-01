"""
Gera embeddings batch para o catalogo de solucoes piloto.

Saidas:
  - data/solucoes_piloto_chunks.json
  - data/solucoes_faiss_index/
"""

from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from bbsia.domain.catalogo.service import OUTPUT_CHUNKS_FILE, materialize_solution_chunks  # noqa: E402
from bbsia.rag.ingestion.embedding import run_embedding  # noqa: E402


def main() -> None:
    materialized = materialize_solution_chunks()
    embedding_result = run_embedding(
        chunks_file=str(OUTPUT_CHUNKS_FILE),
        index_dir="data/solucoes_qdrant_metadata",
        collection_name="bbsia_solucoes",
    )
    print(json.dumps({"catalogo": materialized, "embedding": embedding_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
