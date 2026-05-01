from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("RAG_STRICT_DENSE_ERRORS", "true")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from bbsia.rag.retrieval.retriever import DATA_DIR, search  # noqa: E402
from bbsia.infrastructure.vector_store import COLLECTION_NAME, get_local_qdrant_client  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_diagnostics() -> dict:
    data_dir = Path(DATA_DIR)
    metadata_path = data_dir / "qdrant_index_metadata" / "metadata.json"
    manifest_path = data_dir / "qdrant_index_metadata" / "manifest.json"

    checks: dict[str, object] = {
        "metadata_exists": metadata_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "collection": COLLECTION_NAME,
    }

    if not metadata_path.exists():
        raise RuntimeError(f"Metadata oficial Qdrant ausente: {metadata_path}")
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest oficial Qdrant ausente: {manifest_path}")

    metadata = _load_json(metadata_path)
    metadata_total = int(metadata.get("total_chunks") or len(metadata.get("chunks") or []))
    checks["metadata_total_chunks"] = metadata_total

    client = get_local_qdrant_client(DATA_DIR)
    try:
        collection_exists = bool(client.collection_exists(COLLECTION_NAME))
        checks["collection_exists"] = collection_exists
        if not collection_exists:
            raise RuntimeError(f"Colecao Qdrant ausente: {COLLECTION_NAME}")

        point_count = int(client.count(collection_name=COLLECTION_NAME, exact=True).count)
        checks["point_count"] = point_count
        checks["count_matches_metadata"] = point_count == metadata_total
        if point_count <= 0:
            raise RuntimeError(f"Colecao Qdrant vazia: {COLLECTION_NAME}")
        if point_count != metadata_total:
            raise RuntimeError(
                f"Contagem Qdrant inconsistente: pontos={point_count}; metadata_total_chunks={metadata_total}"
            )
    finally:
        try:
            client.close()
        except Exception:
            pass

    query = "Como funciona o pipeline RAG do chatbot?"
    results = search(query=query, top_k=5, filtro_area="ia")
    dense_scores = [float(item.get("score_dense", 0.0) or 0.0) for item in results]
    sparse_scores = [float(item.get("score_sparse", 0.0) or 0.0) for item in results]

    checks["query"] = query
    checks["result_count"] = len(results)
    checks["dense_scores"] = dense_scores
    checks["sparse_scores"] = sparse_scores
    checks["dense_positive"] = any(score > 0 for score in dense_scores)
    checks["sparse_present"] = any(score > 0 for score in sparse_scores)
    checks["top_docs"] = [item.get("documento") for item in results]

    if not checks["dense_positive"]:
        raise RuntimeError("Busca de diagnostico nao retornou score_dense > 0.")

    return {"status": "pass", "checks": checks}


def main() -> None:
    try:
        payload = run_diagnostics()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "fail", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
