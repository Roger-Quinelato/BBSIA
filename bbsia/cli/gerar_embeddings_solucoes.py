"""
Gera embeddings batch para o catalogo de solucoes piloto.

Saidas:
  - data/solucoes_piloto_chunks.json
  - data/solucoes_faiss_index/
"""

from __future__ import annotations

import json

from bbsia.domain.catalogo.service import OUTPUT_CHUNKS_FILE, materialize_solution_chunks
from bbsia.infrastructure.vector_store import solution_collection_name
from bbsia.rag.ingestion.embedding import run_embedding


def main() -> None:
    materialized = materialize_solution_chunks()
    embedding_result = run_embedding(
        chunks_file=str(OUTPUT_CHUNKS_FILE),
        index_dir="data/solucoes_qdrant_metadata",
        collection_name=solution_collection_name(),
    )
    print(json.dumps({"catalogo": materialized, "embedding": embedding_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
