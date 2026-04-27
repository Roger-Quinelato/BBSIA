"""
Motor RAG para o projeto BBSIA.

Arquitetura:
  - Dense retrieval (FAISS + embeddings)
  - Sparse retrieval (BM25 local)
  - Fusão híbrida (Reciprocal Rank Fusion)
  - Re-ranking opcional com cross-encoder
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import requests
from config import get_env_bool, get_env_int, get_env_list, get_env_str

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale faiss-cpu para usar rag_engine.py") from exc

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar rag_engine.py") from exc


EMBEDDING_MODEL_FALLBACK = get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EXPECTED_EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 1024, min_value=1, max_value=8192)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
OLLAMA_URL = get_env_str("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = get_env_str("DEFAULT_MODEL", "qwen3.5:7b-instruct")
ALLOWED_LLM_MODELS = set(get_env_list("ALLOWED_LLM_MODELS", [DEFAULT_LLM_MODEL]))
ALLOW_REMOTE_OLLAMA = get_env_bool("ALLOW_REMOTE_OLLAMA", False)
DEFAULT_TOP_K = get_env_int("TOP_K", 5, min_value=1, max_value=20)
GENERAL_AREAS = ["tecnologia", "ia", "saude", "infraestrutura", "juridico"]

MAX_CONTEXT_CHUNKS = get_env_int("MAX_CONTEXT_CHUNKS", 3, min_value=1, max_value=10)
MAX_CHARS_PER_CHUNK = get_env_int("MAX_CHARS_PER_CHUNK", 700, min_value=200, max_value=4000)
OLLAMA_TIMEOUT_SEC = get_env_int("OLLAMA_TIMEOUT_SEC", 300, min_value=30, max_value=3600)
OLLAMA_NUM_PREDICT = get_env_int("OLLAMA_NUM_PREDICT", 120, min_value=32, max_value=512)
OLLAMA_NUM_CTX = get_env_int("OLLAMA_NUM_CTX", 8192, min_value=512, max_value=131072)
MIN_DENSE_SCORE_FOR_ANSWER = get_env_int("MIN_DENSE_SCORE_PERCENT", 18, min_value=0, max_value=100) / 100.0

HYBRID_DENSE_CANDIDATES = get_env_int("HYBRID_DENSE_CANDIDATES", 40, min_value=5, max_value=2000)
HYBRID_SPARSE_CANDIDATES = get_env_int("HYBRID_SPARSE_CANDIDATES", 80, min_value=5, max_value=5000)
RRF_K = get_env_int("RRF_K", 60, min_value=1, max_value=200)

ENABLE_RERANKER = get_env_bool("ENABLE_RERANKER", True)
PRELOAD_RAG_ON_STARTUP = get_env_bool("PRELOAD_RAG_ON_STARTUP", True)
PRELOAD_RERANKER_ON_STARTUP = get_env_bool("PRELOAD_RERANKER_ON_STARTUP", False)
RERANKER_MODEL = get_env_str("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATES = get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200)
RERANKER_TOP_N = get_env_int("RERANKER_TOP_N", 3, min_value=1, max_value=10)
RERANKER_MAX_LENGTH = get_env_int("RERANKER_MAX_LENGTH", 512, min_value=128, max_value=2048)
HF_LOCAL_FILES_ONLY = get_env_bool("HF_LOCAL_FILES_ONLY", True)
E5_QUERY_PREFIX = "query: "
NO_EVIDENCE_RESPONSE = "Não encontrei evidências suficientes nos documentos indexados."

SYSTEM_PROMPT = (
    "Você é o assistente técnico do BBSIA. Responda EXCLUSIVAMENTE com base no contexto "
    "fornecido. Se a resposta não estiver no contexto, responda: 'Não encontrei evidências "
    "suficientes nos documentos indexados.' Para cada afirmação, adicione a citação no formato "
    "(Sobrenome, Ano)."
)

_CACHE: dict = {}
_CACHE_STATUS: dict = {
    "loaded_at_utc": None,
    "last_error": None,
    "last_preload_at_utc": None,
    "last_preload_error": None,
}


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_ollama_endpoint() -> None:
    parsed = urlparse(OLLAMA_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("OLLAMA_URL invalida.")
    if not ALLOW_REMOTE_OLLAMA and not _is_loopback_host(parsed.hostname):
        raise RuntimeError("OLLAMA_URL deve apontar para loopback/local quando ALLOW_REMOTE_OLLAMA=false.")


def validate_ollama_model(model: str) -> str:
    selected = (model or "").strip()
    if not selected:
        selected = DEFAULT_LLM_MODEL
    if selected not in ALLOWED_LLM_MODELS:
        allowed = ", ".join(sorted(ALLOWED_LLM_MODELS))
        raise ValueError(f"Modelo Ollama nao permitido: {selected}. Permitidos: {allowed}")
    return selected


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]


def _norm(value: str) -> str:
    return value.strip().lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w\-]{2,}\b", (text or "").lower())


def _format_query_for_embedding(query: str) -> str:
    """Aplica o prefixo recomendado para modelos E5 em consultas."""
    return f"{E5_QUERY_PREFIX}{(query or '').strip()}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_index_manifest(index_path: str, metadata_path: str, manifest_path: str) -> None:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest de integridade nao encontrado em: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest de integridade invalido.")

    expected = {
        INDEX_FILE: manifest.get(INDEX_FILE),
        METADATA_FILE: manifest.get(METADATA_FILE),
    }
    actual = {
        INDEX_FILE: _sha256_file(index_path),
        METADATA_FILE: _sha256_file(metadata_path),
    }
    mismatches = [name for name, digest in expected.items() if not digest or digest != actual[name]]
    if mismatches:
        raise ValueError(f"Falha de integridade no indice FAISS: {', '.join(mismatches)}")


def _build_sparse_index(chunks: list[dict]) -> tuple[list[Counter], list[int], dict[str, int], float]:
    token_counts: list[Counter] = []
    doc_lengths: list[int] = []
    doc_freq: defaultdict[str, int] = defaultdict(int)

    for item in chunks:
        tokens = _tokenize(str(item.get("texto", "")))
        counter = Counter(tokens)
        token_counts.append(counter)
        length = int(sum(counter.values()))
        doc_lengths.append(length)
        for token in counter:
            doc_freq[token] += 1

    avgdl = float(sum(doc_lengths) / max(len(doc_lengths), 1))
    return token_counts, doc_lengths, dict(doc_freq), avgdl


def _load_resources() -> dict:
    if _CACHE:
        return _CACHE

    base_dir = _script_dir()
    index_path = os.path.join(base_dir, INDEX_DIR, INDEX_FILE)
    metadata_path = os.path.join(base_dir, INDEX_DIR, METADATA_FILE)
    manifest_path = os.path.join(base_dir, INDEX_DIR, MANIFEST_FILE)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice FAISS nao encontrado em: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

    _verify_index_manifest(index_path, metadata_path, manifest_path)
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if isinstance(metadata, dict) and "chunks" in metadata:
        chunks = metadata["chunks"]
        embedding_model = metadata.get("model_name", EMBEDDING_MODEL_FALLBACK)
    elif isinstance(metadata, list):
        chunks = metadata
        embedding_model = EMBEDDING_MODEL_FALLBACK
    else:
        raise ValueError("Formato de metadata invalido.")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Metadata nao possui chunks validos.")

    metadata_dim = int(metadata.get("embedding_dim", 0)) if isinstance(metadata, dict) else 0
    index_dim = int(index.d)
    if metadata_dim and metadata_dim != index_dim:
        raise ValueError(
            "Metadata e indice FAISS estao inconsistentes: "
            f"metadata={metadata_dim}, indice={index_dim}. Execute /reprocessar."
        )
    if index_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Indice FAISS incompativel com o modelo de embeddings atual. "
            f"Indice={index_dim} dimensoes; esperado={EXPECTED_EMBEDDING_DIM}. "
            "Execute /reprocessar para recriar o indice com intfloat/multilingual-e5-large."
        )

    try:
        model = SentenceTransformer(embedding_model, local_files_only=HF_LOCAL_FILES_ONLY)
    except Exception as exc:
        raise RuntimeError(
            f"Modelo de embeddings local '{embedding_model}' nao esta disponivel. "
            "Pre-carregue o modelo no cache/local path ou defina HF_LOCAL_FILES_ONLY=false "
            "apenas durante preparacao do ambiente."
        ) from exc
    embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    token_counts, doc_lengths, doc_freq, avgdl = _build_sparse_index(chunks)

    _CACHE.update(
        {
            "index": index,
            "chunks": chunks,
            "model": model,
            "embeddings": embeddings,
            "embedding_model": embedding_model,
            "token_counts": token_counts,
            "doc_lengths": doc_lengths,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "reranker": None,
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _CACHE_STATUS["loaded_at_utc"] = _CACHE["loaded_at_utc"]
    _CACHE_STATUS["last_error"] = None
    return _CACHE


def _get_reranker() -> CrossEncoder | None:
    if not ENABLE_RERANKER:
        return None

    data = _load_resources()
    reranker = data.get("reranker")
    if reranker is None:
        try:
            reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=RERANKER_MAX_LENGTH,
                local_files_only=HF_LOCAL_FILES_ONLY,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Re-ranker local '{RERANKER_MODEL}' nao esta disponivel. "
                "Baixe/cache o modelo no ambiente air-gapped ou defina ENABLE_RERANKER=false."
            ) from exc
        data["reranker"] = reranker
    return reranker


def reload_resources() -> None:
    _CACHE.clear()
    _CACHE_STATUS["loaded_at_utc"] = None


def preload_resources(load_reranker: bool = PRELOAD_RERANKER_ON_STARTUP) -> dict:
    """Carrega indice, modelo de embeddings e, opcionalmente, re-ranker no cache."""
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        data = _load_resources()
        reranker_loaded = False
        if load_reranker and ENABLE_RERANKER:
            reranker_loaded = _get_reranker() is not None
        _CACHE_STATUS["last_preload_at_utc"] = started_at
        _CACHE_STATUS["last_preload_error"] = None
        return {
            "status": "ok",
            "preloaded_at_utc": started_at,
            "embedding_model_loaded": bool(data.get("model")),
            "reranker_loaded": reranker_loaded or bool(data.get("reranker")),
            "total_chunks": len(data.get("chunks", [])),
            "embedding_model": data.get("embedding_model"),
        }
    except Exception as exc:
        message = str(exc)
        _CACHE_STATUS["last_error"] = message
        _CACHE_STATUS["last_preload_at_utc"] = started_at
        _CACHE_STATUS["last_preload_error"] = message
        raise


def cache_health(load_if_empty: bool = False) -> dict:
    """Retorna um snapshot leve do cache/modelos sem obrigatoriamente carregar recursos."""
    if load_if_empty and not _CACHE:
        _load_resources()

    chunks = _CACHE.get("chunks") or []
    index = _CACHE.get("index")
    embeddings = _CACHE.get("embeddings")
    embedding_dim = int(getattr(index, "d", 0) or 0) if index is not None else None
    embedding_count = int(getattr(index, "ntotal", 0) or 0) if index is not None else 0

    return {
        "resources_cached": bool(_CACHE),
        "embedding_model_loaded": bool(_CACHE.get("model")),
        "reranker_enabled": ENABLE_RERANKER,
        "reranker_cached": bool(_CACHE.get("reranker")),
        "preload_rag_on_startup": PRELOAD_RAG_ON_STARTUP,
        "preload_reranker_on_startup": PRELOAD_RERANKER_ON_STARTUP,
        "hf_local_files_only": HF_LOCAL_FILES_ONLY,
        "embedding_model": _CACHE.get("embedding_model") or EMBEDDING_MODEL_FALLBACK,
        "reranker_model": RERANKER_MODEL if ENABLE_RERANKER else None,
        "total_chunks": len(chunks),
        "faiss_vectors": embedding_count,
        "embedding_dim": embedding_dim,
        "embeddings_matrix_shape": list(embeddings.shape) if isinstance(embeddings, np.ndarray) else None,
        "loaded_at_utc": _CACHE.get("loaded_at_utc") or _CACHE_STATUS.get("loaded_at_utc"),
        "last_error": _CACHE_STATUS.get("last_error"),
        "last_preload_at_utc": _CACHE_STATUS.get("last_preload_at_utc"),
        "last_preload_error": _CACHE_STATUS.get("last_preload_error"),
        "min_dense_score_percent": int(MIN_DENSE_SCORE_FOR_ANSWER * 100),
        "min_dense_score": MIN_DENSE_SCORE_FOR_ANSWER,
    }


def list_available_areas() -> list[str]:
    data = _load_resources()
    areas = {c.get("area", "geral") for c in data["chunks"]}
    merged = list(GENERAL_AREAS)
    extras = sorted(area for area in areas if area not in GENERAL_AREAS)
    return merged + extras


def list_available_assuntos() -> list[str]:
    data = _load_resources()
    assuntos = set()
    for c in data["chunks"]:
        for assunto in c.get("assuntos", []):
            if isinstance(assunto, str) and assunto.strip():
                assuntos.add(assunto)
    return sorted(assuntos)


def list_ollama_models(timeout_sec: int = 5) -> list[str]:
    validate_ollama_endpoint()
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout_sec)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = sorted({m.get("name") for m in models if m.get("name")} & ALLOWED_LLM_MODELS)
        return names or sorted(ALLOWED_LLM_MODELS)
    except Exception:
        return sorted(ALLOWED_LLM_MODELS)


def _filter_ids(
    chunks: list[dict],
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
) -> list[int]:
    areas = {_norm(v) for v in _as_list(filtro_area) if v.strip()}
    assuntos = {_norm(v) for v in _as_list(filtro_assunto) if v.strip()}

    if not areas and not assuntos:
        return list(range(len(chunks)))

    eligible: list[int] = []
    for idx, chunk in enumerate(chunks):
        area_val = _norm(chunk.get("area", ""))
        assunto_vals = {_norm(a) for a in chunk.get("assuntos", []) if isinstance(a, str)}

        area_ok = not areas or area_val in areas
        assunto_ok = not assuntos or bool(assunto_vals.intersection(assuntos))

        if area_ok and assunto_ok:
            eligible.append(idx)

    return eligible


def _dense_ranked_candidates(
    query_vec: np.ndarray,
    eligible_ids: list[int],
    chunks_len: int,
    index: faiss.Index,
    all_embeddings: np.ndarray,
    top_n: int,
) -> tuple[list[int], dict[int, float]]:
    top_n = max(1, min(top_n, len(eligible_ids)))

    # Caminho rapido: sem filtros extras.
    if len(eligible_ids) == chunks_len:
        distances, indices = index.search(np.array([query_vec], dtype=np.float32), min(top_n, index.ntotal))
        ranked: list[int] = []
        score_map: dict[int, float] = {}
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = int(idx)
            ranked.append(doc_id)
            score_map[doc_id] = float(score)
        return ranked, score_map

    # Com filtros: calcula similaridade no subconjunto.
    eligible_embeddings = all_embeddings[eligible_ids]
    scores = np.dot(eligible_embeddings, query_vec)

    if top_n < len(scores):
        top_pos = np.argpartition(-scores, top_n - 1)[:top_n]
    else:
        top_pos = np.arange(len(scores))
    top_pos = top_pos[np.argsort(-scores[top_pos])]

    ranked = [eligible_ids[int(pos)] for pos in top_pos]
    score_map = {eligible_ids[int(pos)]: float(scores[int(pos)]) for pos in top_pos}
    return ranked, score_map


def _bm25_score(
    query_tokens: list[str],
    doc_counter: Counter,
    doc_len: int,
    doc_count: int,
    doc_freq: dict[str, int],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or doc_len <= 0:
        return 0.0

    score = 0.0
    for token in query_tokens:
        tf = doc_counter.get(token, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(token, 0)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1e-6)))
        score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-6))
    return float(score)


def _sparse_ranked_candidates(
    query: str,
    eligible_ids: list[int],
    token_counts: list[Counter],
    doc_lengths: list[int],
    doc_freq: dict[str, int],
    avgdl: float,
    top_n: int,
) -> tuple[list[int], dict[int, float]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [], {}

    doc_count = len(token_counts)
    scores: list[tuple[int, float]] = []
    for doc_id in eligible_ids:
        score = _bm25_score(
            query_tokens=query_tokens,
            doc_counter=token_counts[doc_id],
            doc_len=doc_lengths[doc_id],
            doc_count=doc_count,
            doc_freq=doc_freq,
            avgdl=avgdl,
        )
        if score > 0:
            scores.append((doc_id, score))

    if not scores:
        return [], {}

    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[: max(1, min(top_n, len(scores)))]
    return [doc_id for doc_id, _ in scores], {doc_id: float(score) for doc_id, score in scores}


def _fuse_rankings(rankings: list[list[int]], rrf_k: int = RRF_K) -> dict[int, float]:
    fused: defaultdict[int, float] = defaultdict(float)
    for ranked in rankings:
        for rank_pos, doc_id in enumerate(ranked):
            fused[doc_id] += 1.0 / (rrf_k + rank_pos + 1)
    return dict(fused)


def _rerank_candidates(
    query: str,
    candidate_ids: list[int],
    chunks: list[dict],
) -> tuple[list[int], dict[int, float]]:
    reranker = _get_reranker()
    if reranker is None or not candidate_ids:
        return candidate_ids, {}

    pairs = [
        (query, str(chunks[doc_id].get("texto", ""))[:1200])
        for doc_id in candidate_ids
    ]
    scores = reranker.predict(pairs)
    scored = list(zip(candidate_ids, [float(v) for v in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored], {doc_id: score for doc_id, score in scored}


def _dedupe_by_parent(candidate_ids: list[int], chunks: list[dict], limit: int) -> list[int]:
    """Evita enviar varios child chunks do mesmo parent para o contexto final."""
    selected: list[int] = []
    seen: set[str] = set()

    for doc_id in candidate_ids:
        chunk = chunks[doc_id]
        key = str(chunk.get("parent_id") or f"chunk-{doc_id}")
        if key in seen:
            continue
        seen.add(key)
        selected.append(doc_id)
        if len(selected) >= limit:
            break

    return selected


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    data = _load_resources()
    chunks = data["chunks"]
    model = data["model"]
    index = data["index"]
    all_embeddings = data["embeddings"]
    token_counts = data["token_counts"]
    doc_lengths = data["doc_lengths"]
    doc_freq = data["doc_freq"]
    avgdl = data["avgdl"]

    top_k = max(1, int(top_k))
    eligible_ids = _filter_ids(chunks, filtro_area=filtro_area, filtro_assunto=filtro_assunto)
    if not eligible_ids:
        return []

    query_vec = model.encode(
        [_format_query_for_embedding(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    if int(query_vec.shape[0]) != int(index.d):
        raise ValueError(
            "Dimensao da consulta incompativel com o indice FAISS. "
            f"Consulta={int(query_vec.shape[0])}; indice={int(index.d)}. "
            "Execute /reprocessar apos alterar o modelo de embeddings."
        )

    dense_n = max(top_k * 4, HYBRID_DENSE_CANDIDATES)
    sparse_n = max(top_k * 6, HYBRID_SPARSE_CANDIDATES)

    dense_ranked, dense_scores = _dense_ranked_candidates(
        query_vec=query_vec,
        eligible_ids=eligible_ids,
        chunks_len=len(chunks),
        index=index,
        all_embeddings=all_embeddings,
        top_n=dense_n,
    )
    sparse_ranked, sparse_scores = _sparse_ranked_candidates(
        query=query,
        eligible_ids=eligible_ids,
        token_counts=token_counts,
        doc_lengths=doc_lengths,
        doc_freq=doc_freq,
        avgdl=avgdl,
        top_n=sparse_n,
    )

    fused_scores = _fuse_rankings([dense_ranked, sparse_ranked], rrf_k=RRF_K)
    if not fused_scores:
        fused_scores = {doc_id: score for doc_id, score in dense_scores.items()}

    candidate_pool = max(top_k * 3, RERANKER_CANDIDATES if ENABLE_RERANKER else top_k)
    candidate_ids = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    candidate_ids = candidate_ids[: max(1, min(candidate_pool, len(candidate_ids)))]

    reranked_ids, rerank_scores = _rerank_candidates(query=query, candidate_ids=candidate_ids, chunks=chunks)
    final_ids = _dedupe_by_parent(reranked_ids, chunks, top_k)

    results: list[dict] = []
    for doc_id in final_ids:
        meta = chunks[doc_id]
        score_final = rerank_scores.get(doc_id, fused_scores.get(doc_id, dense_scores.get(doc_id, 0.0)))
        results.append(
            {
                "score": float(score_final),
                "id": meta.get("id", doc_id),
                "documento": meta.get("documento", "desconhecido"),
                "pagina": meta.get("pagina"),
                "area": meta.get("area", "geral"),
                "assuntos": meta.get("assuntos", []),
                "doc_titulo": meta.get("doc_titulo", ""),
                "doc_autores": meta.get("doc_autores", []),
                "doc_ano": meta.get("doc_ano"),
                "section_heading": meta.get("section_heading"),
                "content_type": meta.get("content_type", "text"),
                "parent_id": meta.get("parent_id"),
                "parent_text": meta.get("parent_text"),
                "ocr_usado": bool(meta.get("ocr_usado", False)),
                "table_index": meta.get("table_index"),
                "texto": meta.get("texto", ""),
                "chunk_index": meta.get("chunk_index"),
                "score_dense": float(dense_scores.get(doc_id, 0.0)),
                "score_sparse": float(sparse_scores.get(doc_id, 0.0)),
                "score_rerank": float(rerank_scores.get(doc_id, 0.0)),
            }
        )

    return results


def _format_source_label(item: dict) -> str:
    """Gera rótulo acadêmico: 'Sobrenome, Ano — Título' quando disponível."""
    autores = item.get("doc_autores", [])
    ano = item.get("doc_ano")
    titulo = item.get("doc_titulo", "")
    documento = item.get("documento", "desconhecido")

    if autores and isinstance(autores, list) and autores[0]:
        # Usa último sobrenome do primeiro autor
        primeiro_autor = autores[0].strip()
        partes_nome = primeiro_autor.split()
        sobrenome = partes_nome[-1] if partes_nome else primeiro_autor
        label = sobrenome
        if ano:
            label += f", {ano}"
        if titulo:
            label += f' — "{titulo}"'
        return label

    # Fallback: nome do arquivo sem extensão
    import os
    nome_base = os.path.splitext(os.path.basename(documento))[0]
    if ano:
        return f"{nome_base} ({ano})"
    return nome_base


def _format_citation_label(item: dict) -> str:
    autores = item.get("doc_autores", [])
    ano = item.get("doc_ano") or "s.d."

    if autores and isinstance(autores, list) and autores[0]:
        primeiro_autor = str(autores[0]).strip()
        partes_nome = primeiro_autor.split()
        sobrenome = partes_nome[-1] if partes_nome else primeiro_autor
    else:
        documento = str(item.get("documento", "documento"))
        sobrenome = os.path.splitext(os.path.basename(documento))[0]

    sobrenome = re.sub(r"\s+", " ", sobrenome).strip() or "Documento"
    return f"{sobrenome}, {ano}"


def _build_context(results: list[dict]) -> str:
    parts = []
    for i, item in enumerate(results, start=1):
        trecho = str(item.get("parent_text") or item.get("texto", ""))
        if len(trecho) > MAX_CHARS_PER_CHUNK:
            trecho = trecho[:MAX_CHARS_PER_CHUNK].rstrip() + "..."
        section = item.get("section_heading") or "nao informado"
        content_type = item.get("content_type") or "text"
        source_label = _format_source_label(item)
        citation_label = _format_citation_label(item)
        parts.append(
            "\n".join(
                [
                    f"Fonte {i}: {source_label} (p. {item.get('pagina', '?')})",
                    f"Citacao obrigatoria: ({citation_label})",
                    f"Documento: {item.get('documento')}",
                    f"Pagina: {item.get('pagina')}",
                    f"Secao: {section}",
                    f"Tipo: {content_type}",
                    f"Area: {item.get('area')}",
                    f"Assuntos: {', '.join(item.get('assuntos', []))}",
                    f"Trecho: {trecho}",
                ]
            )
        )
    return "\n\n".join(parts)


def build_prompt(pergunta: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "--- CONTEXTO FORNECIDO ---\n"
        f"{context}\n\n"
        "--- PERGUNTA ---\n"
        f"{pergunta}\n\n"
        "--- RESPOSTA ---\n"
    )


def query_ollama(prompt: str, model: str = DEFAULT_LLM_MODEL, timeout_sec: int = OLLAMA_TIMEOUT_SEC) -> str:
    validate_ollama_endpoint()
    model = validate_ollama_model(model)
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        },
        stream=True,
        timeout=(10, timeout_sec),
    ) as response:
        response.raise_for_status()

        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = payload.get("response")
            if piece:
                parts.append(piece)
            if payload.get("done"):
                break

    merged = "".join(parts).strip()
    if not merged:
        raise RuntimeError("Ollama retornou resposta vazia.")
    return merged


def _unique_sources(results: list[dict]) -> list[str]:
    seen = set()
    sources = []
    for r in results:
        label = _format_source_label(r)
        src = f"{label} (p. {r.get('pagina', '?')})"
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def _declares_not_found(answer: str) -> bool:
    normalized = (answer or "").lower()
    markers = [
        "não encontrei evidências suficientes nos documentos indexados",
        "nao encontrei evidencias suficientes nos documentos indexados",
        "nao encontrei",
        "não encontrei",
        "nao foi encontrado",
        "não foi encontrado",
        "suporte suficiente",
        "nao ha informacao",
        "não há informação",
    ]
    return any(marker in normalized for marker in markers)


def _citation_labels(answer: str) -> set[str]:
    pattern = r"\(([^\(\),]{2,80}),\s*((?:19|20)\d{2}|s\.d\.)\)"
    return {
        f"{author.strip()}, {year.strip()}"
        for author, year in re.findall(pattern, answer or "", flags=re.IGNORECASE)
    }


def _faithfulness_check(answer: str, context_results: list[dict]) -> tuple[bool, str | None]:
    """
    Checagem heuristica de grounding.

    Nao substitui avaliacao RAGAS/DeepEval, mas impede respostas sem citacao
    e citacoes que nao existem no contexto enviado ao modelo.
    """
    answer = (answer or "").strip()
    if not answer:
        return False, "resposta vazia"
    if _declares_not_found(answer):
        return True, None

    citations = _citation_labels(answer)
    if not citations:
        return False, "resposta sem citacoes no formato (Sobrenome, Ano)"

    valid_sources = {_format_citation_label(item) for item in context_results}
    invalid = sorted(citations - valid_sources)
    if invalid:
        return False, f"citacoes fora do contexto: {invalid}"

    factual_lines = [
        line.strip()
        for line in re.split(r"[\n]+", answer)
        if len(line.strip().split()) >= 8 and not _declares_not_found(line)
    ]
    uncited_lines = [line for line in factual_lines if not _citation_labels(line)]
    if uncited_lines:
        return False, "ha afirmacoes factuais sem citacao"

    return True, None


def _retrieval_has_answer_signal(results: list[dict]) -> bool:
    if not results:
        return False
    best = results[0]
    if "score_dense" not in best and "score_sparse" not in best:
        return float(best.get("score", 0.0)) > 0.0
    dense = float(best.get("score_dense", 0.0))
    sparse = float(best.get("score_sparse", 0.0))
    return dense >= MIN_DENSE_SCORE_FOR_ANSWER or sparse > 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def evaluate_retrieval_quality(
    query_specs: list[dict],
    top_k: int = DEFAULT_TOP_K,
    search_fn=None,
) -> dict:
    """Executa queries esperadas e mede acerto por area/documento no top-k."""
    runner = search_fn or search
    cases: list[dict] = []
    passed = 0

    for spec in query_specs:
        query = str(spec.get("query", "")).strip()
        expected_area = str(spec.get("area_esperada", "") or "").strip()
        expected_document = str(spec.get("documento_esperado", "") or "").strip()
        expected_any = [str(v).strip() for v in spec.get("documentos_esperados", []) if str(v).strip()]
        if expected_document:
            expected_any.append(expected_document)

        case: dict[str, object] = {
            "query": query,
            "area_esperada": expected_area or None,
            "documentos_esperados": expected_any,
            "passed": False,
        }
        try:
            results = runner(query=query, top_k=top_k)
        except TypeError:
            results = runner(query, top_k)
        except Exception as exc:
            case["erro"] = str(exc)
            cases.append(case)
            continue

        top = results[0] if results else {}
        returned_docs = [str(item.get("documento", "")) for item in results]
        returned_area = str(top.get("area", "")) if top else ""
        area_ok = not expected_area or expected_area == "nenhuma" or returned_area == expected_area
        if expected_area == "nenhuma":
            area_ok = not results or float(top.get("score_dense", top.get("score", 0.0)) or 0.0) < MIN_DENSE_SCORE_FOR_ANSWER

        doc_ok = not expected_any or any(
            expected in returned_doc or returned_doc in expected
            for expected in expected_any
            for returned_doc in returned_docs
        )
        case.update(
            {
                "total_resultados": len(results),
                "area_retornada": returned_area or None,
                "documentos_retornados": returned_docs,
                "score_dense_top1": float(top.get("score_dense", 0.0)) if top else 0.0,
                "score_sparse_top1": float(top.get("score_sparse", 0.0)) if top else 0.0,
                "score_final_top1": float(top.get("score", 0.0)) if top else 0.0,
                "area_ok": area_ok,
                "documento_ok": doc_ok,
            }
        )
        case["passed"] = bool(area_ok and doc_ok)
        if case["passed"]:
            passed += 1
        cases.append(case)

    total = len(query_specs)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "cases": cases,
    }


def calibrate_dense_threshold(
    query_specs: list[dict],
    top_k: int = DEFAULT_TOP_K,
    search_fn=None,
) -> dict:
    """Sugere MIN_DENSE_SCORE_PERCENT com base em queries esperadas."""
    runner = search_fn or search
    results_payload: list[dict] = []
    in_scope_scores: list[float] = []
    out_scope_scores: list[float] = []

    for spec in query_specs:
        query = str(spec.get("query", "")).strip()
        expected_area = str(spec.get("area_esperada", "") or "").strip()
        try:
            try:
                results = runner(query=query, top_k=top_k)
            except TypeError:
                results = runner(query, top_k)
        except Exception as exc:
            results_payload.append({"query": query, "area_esperada": expected_area, "erro": str(exc)})
            continue

        top = results[0] if results else {}
        dense = float(top.get("score_dense", top.get("score", 0.0)) or 0.0) if top else 0.0
        sparse = float(top.get("score_sparse", 0.0) or 0.0) if top else 0.0
        final = float(top.get("score", 0.0) or 0.0) if top else 0.0
        is_out_scope = expected_area == "nenhuma"
        if is_out_scope:
            out_scope_scores.append(dense)
        elif results:
            in_scope_scores.append(dense)

        results_payload.append(
            {
                "query": query,
                "area_esperada": expected_area,
                "area_retornada": top.get("area") if top else None,
                "area_match": (top.get("area") == expected_area) if top and expected_area and not is_out_scope else None,
                "score_dense": round(dense, 4),
                "score_sparse": round(sparse, 4),
                "score_final": round(final, 4),
                "documento": top.get("documento") if top else None,
                "pagina": top.get("pagina") if top else None,
                "total_resultados": len(results),
            }
        )

    p10 = _percentile(in_scope_scores, 0.10)
    p25 = _percentile(in_scope_scores, 0.25)
    out_max = max(out_scope_scores) if out_scope_scores else 0.0
    suggested = p25
    if out_scope_scores and out_max < p25:
        suggested = max(p10, (out_max + p25) / 2.0)
    suggested_percent = max(1, min(100, int(round(suggested * 100))))

    stats = {
        "total_queries": len(query_specs),
        "queries_com_resultado": len(in_scope_scores) + len(out_scope_scores),
        "in_scope_count": len(in_scope_scores),
        "out_scope_count": len(out_scope_scores),
        "dense_min": round(min(in_scope_scores), 4) if in_scope_scores else 0.0,
        "dense_p10": round(p10, 4),
        "dense_p25": round(p25, 4),
        "dense_mediana": round(_percentile(in_scope_scores, 0.50), 4),
        "dense_media": round(float(sum(in_scope_scores) / len(in_scope_scores)), 4) if in_scope_scores else 0.0,
        "dense_max": round(max(in_scope_scores), 4) if in_scope_scores else 0.0,
        "out_scope_dense_max": round(out_max, 4),
        "threshold_sugerido_percent": suggested_percent,
        "threshold_sugerido_float": round(suggested_percent / 100.0, 4),
    }

    return {
        "estatisticas": stats,
        "qualidade": evaluate_retrieval_quality(query_specs, top_k=top_k, search_fn=runner),
        "resultados": results_payload,
    }


def _extractive_grounded_answer(pergunta: str, results: list[dict], reason: str | None = None) -> str:
    highlights: list[str] = []
    for item in results[:MAX_CONTEXT_CHUNKS]:
        texto = str(item.get("parent_text") or item.get("texto", "")).strip()
        if not texto:
            continue
        resumo = texto.replace("\n", " ").strip()
        if len(resumo) > 320:
            resumo = resumo[:320].rstrip() + "..."
        highlights.append(f"- {resumo} ({_format_citation_label(item)})")

    if not highlights:
        return NO_EVIDENCE_RESPONSE

    prefix = NO_EVIDENCE_RESPONSE
    if reason:
        prefix += f" Controle de fidelidade: {reason}."

    return (
        f"{prefix}\n\n"
        "Trechos mais relevantes recuperados:\n"
        + "\n".join(highlights)
        + f"\n\nPergunta original: {pergunta}"
    )


def _extractive_fallback_answer(pergunta: str, results: list[dict], error: Exception) -> str:
    """
    Gera uma resposta util quando o LLM local falha.
    """
    highlights: list[str] = []
    for item in results[:RERANKER_TOP_N]:
        texto = str(item.get("parent_text") or item.get("texto", "")).strip()
        if not texto:
            continue
        resumo = texto.replace("\n", " ").strip()
        if len(resumo) > 240:
            resumo = resumo[:240].rstrip() + "..."
        src = f"{item.get('documento', 'desconhecido')} (p. {item.get('pagina', '?')})"
        highlights.append(f"- {resumo} ({_format_citation_label(item)}) - {src}")

    if not highlights:
        return (
            "Nao foi possivel consultar o Ollama local e tambem nao ha trechos "
            "suficientes para gerar um resumo de fallback."
        )

    return (
        "Nao foi possivel concluir a geracao pelo Ollama local neste momento. "
        "Abaixo esta um resumo baseado diretamente nos trechos recuperados:\n\n"
        + "\n".join(highlights)
        + f"\n\nPergunta original: {pergunta}\n"
        + f"Detalhe tecnico: {error}"
    )


def answer_question(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
) -> dict:
    results = search(
        query=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    if not results:
        return {
            "resposta": NO_EVIDENCE_RESPONSE,
            "fontes": [],
            "resultados": [],
            "prompt": None,
        }

    if not _retrieval_has_answer_signal(results):
        return {
            "resposta": NO_EVIDENCE_RESPONSE,
            "fontes": _unique_sources(results),
            "resultados": results,
            "prompt": None,
        }

    context_limit = min(MAX_CONTEXT_CHUNKS, RERANKER_TOP_N)
    context_results = results[:context_limit]
    context = _build_context(context_results)
    prompt = build_prompt(pergunta=pergunta, context=context)
    try:
        resposta = query_ollama(prompt=prompt, model=model)
        faithful, reason = _faithfulness_check(resposta, context_results)
        if not faithful:
            resposta = _extractive_grounded_answer(pergunta=pergunta, results=context_results, reason=reason)
    except Exception as exc:
        resposta = _extractive_fallback_answer(pergunta=pergunta, results=results, error=exc)

    return {
        "resposta": resposta,
        "fontes": _unique_sources(results),
        "resultados": results,
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa consulta RAG no acervo BBSIA.")
    parser.add_argument("--pergunta", required=True, help="Pergunta a ser respondida")
    parser.add_argument("--modelo", default=DEFAULT_LLM_MODEL, help="Modelo Ollama")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Quantidade de chunks")
    parser.add_argument("--area", action="append", default=[], help="Filtro de area (pode repetir)")
    parser.add_argument("--assunto", action="append", default=[], help="Filtro de assunto (pode repetir)")
    args = parser.parse_args()

    result = answer_question(
        pergunta=args.pergunta,
        model=args.modelo,
        top_k=args.top_k,
        filtro_area=args.area,
        filtro_assunto=args.assunto,
    )

    print("\nResposta:\n")
    print(result["resposta"])
    print("\nFontes:")
    if result["fontes"]:
        for f in result["fontes"]:
            print(f"- {f}")
    else:
        print("- Nenhuma")


if __name__ == "__main__":
    main()
