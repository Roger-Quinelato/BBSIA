"""
Fase 3 do pipeline RAG BBSIA:
gera embeddings dos chunks e cria um indice FAISS persistente.

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
from config import get_env_bool, get_env_int, get_env_str

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale faiss-cpu para usar embedding.py") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar embedding.py") from exc


MODEL_NAME = get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EXPECTED_EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 1024, min_value=1, max_value=8192)
HF_LOCAL_FILES_ONLY = get_env_bool("HF_LOCAL_FILES_ONLY", True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
BATCH_SIZE = 32
E5_PASSAGE_PREFIX = "passage: "
LOGGER = logging.getLogger(__name__)


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


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


def _format_passage_for_embedding(text: str) -> str:
    """Aplica o prefixo recomendado para modelos E5 em documentos."""
    return f"{E5_PASSAGE_PREFIX}{(text or '').strip()}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_index_manifest(index_dir: str, index_path: str, metadata_path: str) -> str:
    manifest_path = os.path.join(index_dir, MANIFEST_FILE)
    manifest = {
        INDEX_FILE: _sha256_file(index_path),
        METADATA_FILE: _sha256_file(metadata_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def run_embedding(
    chunks_file: str = CHUNKS_FILE,
    index_dir: str = INDEX_DIR,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> dict:
    base_dir = _script_dir()
    chunks_path = os.path.join(base_dir, chunks_file)
    index_path = os.path.join(base_dir, index_dir, INDEX_FILE)
    metadata_path = os.path.join(base_dir, index_dir, METADATA_FILE)

    chunks = _load_chunks(chunks_path)
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
    collection_name = "bbsia_chunks"
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload=chunk
            )
        )
    
    batch_size_points = 500
    for j in range(0, len(points), batch_size_points):
        client.upload_points(collection_name=collection_name, points=points[j:j+batch_size_points])

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        f.write("QDRANT_MIGRATED")

    metadata_payload = {
        "model_name": model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "embedding_input_format": "e5_passage_prefix",
        "chunks": chunks,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)

    manifest_path = _write_index_manifest(os.path.dirname(index_path), index_path, metadata_path)

    LOGGER.info(
        "event=embedding_completed total_chunks=%s dim=%s index_path=%s metadata_path=%s manifest_path=%s",
        len(chunks),
        dim,
        index_path,
        metadata_path,
        manifest_path,
    )

    return {
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "index_path": index_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    run_embedding()
