"""
Fase 3 do pipeline RAG BBSIA:
gera embeddings dos chunks e indexa no Qdrant local.

Uso:
  python embedding.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from bbsia.core.config import get_env_bool, get_env_int, get_env_str

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar embedding.py") from exc


MODEL_NAME = get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EXPECTED_EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 1024, min_value=1, max_value=8192)
HF_LOCAL_FILES_ONLY = get_env_bool("HF_LOCAL_FILES_ONLY", True)
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
PARENTS_FILE = os.path.join(DATA_DIR, "parents.json")
METADATA_DIR = os.path.join(DATA_DIR, "qdrant_index_metadata")
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
BATCH_SIZE = 32
E5_PASSAGE_PREFIX = "passage: "
LOGGER = logging.getLogger(__name__)


def _script_dir() -> str:
    return _REPO_ROOT


def _load_chunks(chunks_path: str) -> list[dict]:
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Arquivo {os.path.basename(chunks_path)} nao encontrado. Rode chunking.py antes."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json esta vazio ou em formato invalido.")

    if not all("texto" in c for c in chunks):
        raise ValueError("Alguns chunks nao possuem o campo 'texto'.")

    return chunks


def _load_parents_map(parents_path: str) -> dict[str, str]:
    if not os.path.exists(parents_path):
        return {}

    with open(parents_path, "r", encoding="utf-8") as f:
        parents = json.load(f)

    if not isinstance(parents, dict):
        return {}

    return {str(parent_id): str(text) for parent_id, text in parents.items()}


def _split_lean_chunks_and_parents(chunks: list[dict], parents_map: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    resolved_parents = dict(parents_map)
    lean_chunks: list[dict] = []

    for chunk in chunks:
        lean_chunk = dict(chunk)
        parent_text = lean_chunk.pop("parent_text", None)
        parent_id = str(lean_chunk.get("parent_id") or "")
        if parent_id and parent_id not in resolved_parents and parent_text:
            resolved_parents[parent_id] = str(parent_text)
        lean_chunks.append(lean_chunk)

    return lean_chunks, resolved_parents


def _format_passage_for_embedding(text: str) -> str:
    """Aplica o prefixo recomendado para modelos E5 em documentos."""
    return f"{E5_PASSAGE_PREFIX}{(text or '').strip()}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_index_manifest(metadata_dir: str, metadata_path: str) -> str:
    manifest_path = os.path.join(metadata_dir, MANIFEST_FILE)
    manifest = {
        "vector_backend": "qdrant_local",
        "metadata_path": os.path.abspath(metadata_path),
        METADATA_FILE: _sha256_file(metadata_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    """Aplica o prefixo recomendado para modelos E5 em documentos."""
    return f"{E5_PASSAGE_PREFIX}{(text or '').strip()}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_index_manifest(metadata_dir: str, metadata_path: str) -> str:
    manifest_path = os.path.join(metadata_dir, MANIFEST_FILE)
    manifest = {
        "vector_backend": "qdrant_local",
        "metadata_path": os.path.abspath(metadata_path),
        METADATA_FILE: _sha256_file(metadata_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_embedding(
    chunks_file: str = CHUNKS_FILE,
    metadata_dir: str = METADATA_DIR,
    index_dir: str | None = None,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    collection_name: str = "bbsia_chunks",
) -> dict:
    if index_dir:
        # Alias legado: manter compatibilidade com chamadores antigos.
        metadata_dir = index_dir

    base_dir = _script_dir()
    chunks_path = os.path.join(base_dir, chunks_file)
    metadata_path = os.path.join(base_dir, metadata_dir, METADATA_FILE)

    chunks = _load_chunks(chunks_path)
    parents_path = os.path.join(base_dir, PARENTS_FILE)
    lean_chunks, parents_map = _split_lean_chunks_and_parents(chunks, _load_parents_map(parents_path))
    texts = [_format_passage_for_embedding(str(c["texto"])) for c in chunks]

    LOGGER.info("event=embedding_model_loading model=%s", model_name)
    try:
        model = SentenceTransformer(model_name, local_files_only=HF_LOCAL_FILES_ONLY)
    except Exception as exc:
        raise RuntimeError(
            f"Modelo de embeddings local '{model_name}' nao esta disponivel. "
            "Pre-carregue o modelo no cache/local path ou defina HF_LOCAL_FILES_ONLY=false "
            "apenas durante preparacao do ambiente."
        ) from exc

    LOGGER.info("event=embedding_encoding_started total_chunks=%s", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    dim = int(embeddings.shape[1])
    if dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Dimensao inesperada do modelo de embeddings. "
            f"Esperado: {EXPECTED_EMBEDDING_DIM}; obtido: {dim}. "
            "Verifique EMBEDDING_MODEL/EMBEDDING_DIM e recrie o indice via /reprocessar."
        )

    LOGGER.info("event=embedding_index_building dim=%s", dim)
    
    qdrant_path = os.path.join(DATA_DIR, "qdrant_db")
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    client = QdrantClient(path=qdrant_path)
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    points = []
    for i, (chunk, emb) in enumerate(zip(lean_chunks, embeddings)):
        payload = dict(chunk)
        parent_id = str(payload.get("parent_id") or "")
        if parent_id in parents_map:
            payload["parent_text"] = parents_map[parent_id]

        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload=payload
            )
        )
    
    batch_size_points = 500
    for j in range(0, len(points), batch_size_points):
        client.upload_points(collection_name=collection_name, points=points[j:j+batch_size_points])
    client.close()

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    metadata_payload = {
        "model_name": model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "embedding_input_format": "e5_passage_prefix",
        "chunks": lean_chunks,
        "parents": parents_map,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)

    manifest_path = _write_index_manifest(os.path.dirname(metadata_path), metadata_path)

    LOGGER.info(
        "event=embedding_completed total_chunks=%s dim=%s metadata_path=%s manifest_path=%s",
        len(chunks),
        dim,
        metadata_path,
        manifest_path,
    )

    return {
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        # Alias de compatibilidade: manter campo historico no payload de retorno.
        "index_path": metadata_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "metadata_dir": os.path.dirname(metadata_path),
        "vector_backend": "qdrant_local",
    }


if __name__ == "__main__":
    run_embedding()
