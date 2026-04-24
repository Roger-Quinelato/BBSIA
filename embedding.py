"""
Fase 3 do pipeline RAG BBSIA:
gera embeddings dos chunks e cria um indice FAISS persistente.

Uso:
  python embedding.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from config import get_env_str

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale faiss-cpu para usar embedding.py") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar embedding.py") from exc


MODEL_NAME = get_env_str("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
BATCH_SIZE = 32


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
    texts = [c["texto"] for c in chunks]

    print(f"Carregando modelo de embedding: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Gerando embeddings para {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    dim = int(embeddings.shape[1])

    print(f"Criando indice FAISS (IndexFlatIP), dim={dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    metadata_payload = {
        "model_name": model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "chunks": chunks,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  EMBEDDING CONCLUIDO")
    print("=" * 60)
    print(f"  Chunks vetorizados : {len(chunks)}")
    print(f"  Dimensao embedding : {dim}")
    print(f"  Indice salvo em    : {index_path}")
    print(f"  Metadata salva em  : {metadata_path}")
    print("=" * 60)

    return {
        "total_chunks": len(chunks),
        "embedding_dim": dim,
        "index_path": index_path,
        "metadata_path": metadata_path,
    }


if __name__ == "__main__":
    run_embedding()
