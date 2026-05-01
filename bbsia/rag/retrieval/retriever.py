from __future__ import annotations
import argparse, atexit, inspect, ipaddress, json, math, os, re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, AsyncGenerator, Any
import httpx
import threading
import numpy as np
import requests

from bbsia.core.config import get_env_bool, get_env_int, get_env_list, get_env_str
from bbsia.rag.retrieval.query_planning import plan_query
from bbsia.rag.shared.sources import _format_citation_label, _format_source_label
from bbsia.infrastructure.vector_store import dense_ranked_candidates, get_local_qdrant_client, vector_store_health, COLLECTION_NAME, COLLECTION_SOLUTIONS

"""
Motor RAG para o projeto BBSIA.

Arquitetura:
  - Dense retrieval (Qdrant local + embeddings)
  - Sparse retrieval (BM25 local)
  - Fusão híbrida (Reciprocal Rank Fusion)
  - Re-ranking opcional com cross-encoder
"""

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar rag_engine.py") from exc

EMBEDDING_MODEL_FALLBACK = get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

EXPECTED_EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 1024, min_value=1, max_value=8192)

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data")

METADATA_DIR = os.path.join(DATA_DIR, "qdrant_index_metadata")
LEGACY_METADATA_DIR = os.path.join(DATA_DIR, "faiss_index")

METADATA_FILE = "metadata.json"

DEFAULT_TOP_K = get_env_int("TOP_K", 5, min_value=1, max_value=20)

GENERAL_AREAS = ["tecnologia", "ia", "saude", "infraestrutura", "juridico"]

MAX_CHARS_PER_CHUNK = get_env_int("MAX_CHARS_PER_CHUNK", 700, min_value=200, max_value=4000)

MIN_DENSE_SCORE_FOR_ANSWER = get_env_int("MIN_DENSE_SCORE_PERCENT", 18, min_value=0, max_value=100) / 100.0

HYBRID_DENSE_CANDIDATES = get_env_int("HYBRID_DENSE_CANDIDATES", 40, min_value=5, max_value=2000)

HYBRID_SPARSE_CANDIDATES = get_env_int("HYBRID_SPARSE_CANDIDATES", 80, min_value=5, max_value=5000)

RRF_K = get_env_int("RRF_K", 60, min_value=1, max_value=200)

PRELOAD_RAG_ON_STARTUP = get_env_bool("PRELOAD_RAG_ON_STARTUP", True)
PRELOAD_RERANKER_ON_STARTUP = get_env_bool("PRELOAD_RERANKER_ON_STARTUP", False)
ENABLE_QUERY_PLANNING = get_env_bool("ENABLE_QUERY_PLANNING", False)
RETRIEVAL_DOMAIN_DOCUMENTS = "documentos"
RETRIEVAL_DOMAIN_SOLUTIONS = "solucoes"
_COLLECTION_BY_RETRIEVAL_DOMAIN = {
    RETRIEVAL_DOMAIN_DOCUMENTS: COLLECTION_NAME,
    RETRIEVAL_DOMAIN_SOLUTIONS: COLLECTION_SOLUTIONS,
}

HF_LOCAL_FILES_ONLY = get_env_bool("HF_LOCAL_FILES_ONLY", True)
E5_QUERY_PREFIX = "query: "

RERANKER_MODEL = get_env_str("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_CANDIDATES = get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200)


def _attach_parent_text(chunks: list[dict], parents_map: dict | None) -> None:
    if not isinstance(parents_map, dict) or not parents_map:
        return

    for chunk in chunks:
        parent_id = str(chunk.get("parent_id") or "")
        if parent_id and "parent_text" not in chunk and parent_id in parents_map:
            chunk["parent_text"] = str(parents_map[parent_id])


class IndexStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}
        self._status = {}

    def _init_status(self, collection: str):
        if collection not in self._status:
            self._status[collection] = {
                "loaded_at_utc": None,
                "last_error": None,
                "last_preload_at_utc": None,
                "last_preload_error": None,
            }

    def get(self, collection: str = COLLECTION_NAME) -> dict:
        with self._lock:
            self._init_status(collection)
            if collection not in self._data:
                self._data[collection] = self._load_from_disk(collection)
                self._status[collection]["loaded_at_utc"] = self._data[collection].get("loaded_at_utc")
                self._status[collection]["last_error"] = None
            return self._data[collection]

    def reload(self, collection: str = COLLECTION_NAME) -> None:
        new_data = self._load_from_disk(collection)
        with self._lock:
            self._init_status(collection)
            self._data[collection] = new_data
            self._status[collection]["loaded_at_utc"] = new_data.get("loaded_at_utc")
            self._status[collection]["last_error"] = None

    def get_status(self, key: str, collection: str = COLLECTION_NAME):
        with self._lock:
            return self._status.get(collection, {}).get(key)
            
    def set_status(self, key: str, value, collection: str = COLLECTION_NAME):
        with self._lock:
            self._init_status(collection)
            self._status[collection][key] = value

    def has_data(self, collection: str = COLLECTION_NAME) -> bool:
        with self._lock:
            return collection in self._data

    def get_data_if_loaded(self, collection: str = COLLECTION_NAME) -> dict | None:
        with self._lock:
            return self._data.get(collection)

    def close(self) -> None:
        with self._lock:
            for col, data in self._data.items():
                qclient = data.get("qclient")
                if qclient is not None and hasattr(qclient, "close"):
                    try:
                        qclient.close()
                    except Exception:
                        pass
            self._data.clear()

    def _load_from_disk(self, collection: str) -> dict:
        import json
        from sentence_transformers import SentenceTransformer
        
        metadata_path = _resolve_metadata_path(collection)

        qclient = get_local_qdrant_client(DATA_DIR)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if isinstance(metadata, dict) and "chunks" in metadata:
            chunks = metadata["chunks"]
            embedding_model = metadata.get("model_name", EMBEDDING_MODEL_FALLBACK)
            parents_map = metadata.get("parents", {})
        elif isinstance(metadata, list):
            chunks = metadata
            embedding_model = EMBEDDING_MODEL_FALLBACK
            parents_map = {}
        else:
            raise ValueError("Formato de metadata invalido.")

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Metadata nao possui chunks validos.")

        _attach_parent_text(chunks, parents_map)

        try:
            model = SentenceTransformer(embedding_model, local_files_only=HF_LOCAL_FILES_ONLY)
        except Exception as exc:
            raise RuntimeError("Modelo indisponivel") from exc
            
        token_counts, doc_lengths, doc_freq, avgdl = _build_sparse_index(chunks)

        return {
            "qclient": qclient,
            "chunks": chunks,
            "model": model,
            "embeddings": None,
            "embedding_model": embedding_model,
            "metadata_path": metadata_path,
            "token_counts": token_counts,
            "doc_lengths": doc_lengths,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "reranker": None,
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

index_store = IndexStore()
atexit.register(index_store.close)

def _script_dir() -> str:
    return _REPO_ROOT


def _accepts_positional_arg(fn) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values()
    )


def _call_with_optional_collection(fn, collection: str):
    if _accepts_positional_arg(fn):
        return fn(collection)
    return fn()


def _call_status_with_optional_collection(fn, key: str, collection: str):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(key, collection)
    accepts_varargs = any(
        parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in signature.parameters.values()
    )
    if accepts_varargs or len(signature.parameters) >= 2:
        return fn(key, collection)
    return fn(key)


def _resolve_metadata_path(collection: str) -> str:
    """Usa metadata oficial nova e aceita fallback legado temporario."""
    if collection == COLLECTION_SOLUTIONS:
        preferred = os.path.join(DATA_DIR, "solucoes_qdrant_metadata", METADATA_FILE)
    else:
        preferred = os.path.join(DATA_DIR, "qdrant_index_metadata", METADATA_FILE)
        
    legacy = os.path.join(DATA_DIR, "faiss_index", METADATA_FILE)
    if os.path.exists(preferred):
        return preferred
    if collection == COLLECTION_NAME and os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(f"Metadata nao encontrada para colecao {collection}")

def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]

def _has_filter_value(value: str | Iterable[str] | None) -> bool:
    return any(v.strip() for v in _as_list(value))

def _norm(value: str) -> str:
    return value.strip().lower()

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w\-]{2,}\b", (text or "").lower())

def _format_query_for_embedding(query: str) -> str:
    """Aplica o prefixo recomendado para modelos E5 em consultas."""
    return f"{E5_QUERY_PREFIX}{(query or '').strip()}"

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

def _load_resources(collection: str = COLLECTION_NAME) -> dict:
    return index_store.get(collection)

def reload_resources(collection: str = COLLECTION_NAME) -> None:
    index_store.reload(collection)

def preload_resources(load_reranker: bool = PRELOAD_RERANKER_ON_STARTUP, collection: str = COLLECTION_NAME) -> dict:
    """Carrega indice, modelo de embeddings e, opcionalmente, re-ranker no cache."""
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        data = _call_with_optional_collection(_load_resources, collection)
        reranker_loaded = False
        if load_reranker and ENABLE_RERANKER:
            reranker_loaded = _get_reranker() is not None
        index_store.set_status("last_preload_at_utc", started_at, collection)
        index_store.set_status("last_preload_error", None, collection)
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
        index_store.set_status("last_error", message, collection)
        index_store.set_status("last_preload_at_utc", started_at, collection)
        index_store.set_status("last_preload_error", message, collection)
        raise

def cache_health(load_if_empty: bool = False, collection: str = COLLECTION_NAME) -> dict:
    """Retorna um snapshot leve do cache/modelos sem obrigatoriamente carregar recursos."""
    if load_if_empty and not _call_with_optional_collection(index_store.has_data, collection):
        _call_with_optional_collection(_load_resources, collection)

    data = _call_with_optional_collection(index_store.get_data_if_loaded, collection) or {}
    chunks = data.get("chunks") or []
    embeddings = data.get("embeddings")
    qclient = data.get("qclient")
    embedding_count = len(chunks) if chunks else 0
    embedding_dim = EXPECTED_EMBEDDING_DIM

    return {
        "resources_cached": _call_with_optional_collection(index_store.has_data, collection),
        "embedding_model_loaded": bool(data.get("model")),
        "reranker_enabled": ENABLE_RERANKER,
        "reranker_cached": bool(data.get("reranker")),
        "preload_rag_on_startup": PRELOAD_RAG_ON_STARTUP,
        "preload_reranker_on_startup": PRELOAD_RERANKER_ON_STARTUP,
        "hf_local_files_only": HF_LOCAL_FILES_ONLY,
        "embedding_model": data.get("embedding_model") or EMBEDDING_MODEL_FALLBACK,
        "reranker_model": RERANKER_MODEL if ENABLE_RERANKER else None,
        "total_chunks": len(chunks),
        "dense_vectors": embedding_count,
        # Alias de compatibilidade (deprecated): manter enquanto clientes migram.
        "faiss_vectors": embedding_count,
        "qdrant_vectors": embedding_count,
        "vector_store": vector_store_health(qclient),
        "embedding_dim": embedding_dim,
        "embeddings_matrix_shape": list(embeddings.shape) if isinstance(embeddings, np.ndarray) else None,
        "loaded_at_utc": data.get("loaded_at_utc") or _call_status_with_optional_collection(index_store.get_status, "loaded_at_utc", collection),
        "last_error": _call_status_with_optional_collection(index_store.get_status, "last_error", collection),
        "last_preload_at_utc": _call_status_with_optional_collection(index_store.get_status, "last_preload_at_utc", collection),
        "last_preload_error": _call_status_with_optional_collection(index_store.get_status, "last_preload_error", collection),
        "min_dense_score_percent": int(MIN_DENSE_SCORE_FOR_ANSWER * 100),
        "min_dense_score": MIN_DENSE_SCORE_FOR_ANSWER,
    }

def list_available_areas(collection: str = COLLECTION_NAME) -> list[str]:
    data = _call_with_optional_collection(_load_resources, collection)
    areas = {c.get("area", "geral") for c in data["chunks"]}
    merged = list(GENERAL_AREAS)
    extras = sorted(area for area in areas if area not in GENERAL_AREAS)
    return merged + extras

def list_available_assuntos(collection: str = COLLECTION_NAME) -> list[str]:
    data = _call_with_optional_collection(_load_resources, collection)
    assuntos = set()
    for c in data["chunks"]:
        for assunto in c.get("assuntos", []):
            if isinstance(assunto, str) and assunto.strip():
                assuntos.add(assunto)
    return sorted(assuntos)

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
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    qclient: Any,
    top_n: int,
    target_collection: str,
) -> tuple[list[int], dict[int, float]]:
    return dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=top_n,
        target_collection=target_collection,
    )

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

from bbsia.rag.retrieval.reranker import ENABLE_RERANKER, _get_reranker

def _search_single_collection(
    query: str,
    top_k: int,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    collection: str,
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    if ENABLE_QUERY_PLANNING:
        plan = plan_query(query)
        if not _has_filter_value(filtro_area) and plan.filtro_area:
            filtro_area = plan.filtro_area
        if not _has_filter_value(filtro_assunto) and plan.filtro_assunto:
            filtro_assunto = plan.filtro_assunto

    data = _call_with_optional_collection(_load_resources, collection)
    chunks = data["chunks"]
    model = data["model"]
    qclient = data["qclient"]
    token_counts = data["token_counts"]
    doc_lengths = data["doc_lengths"]
    doc_freq = data["doc_freq"]
    avgdl = data["avgdl"]

    
    eligible_ids = _filter_ids(chunks, filtro_area=filtro_area, filtro_assunto=filtro_assunto)
    if not eligible_ids:
        return []

    query_vec = model.encode(
        [_format_query_for_embedding(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    if int(query_vec.shape[0]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Dimensao da consulta incompativel. "
            f"Consulta={int(query_vec.shape[0])}; esperado={EXPECTED_EMBEDDING_DIM}. "
        )

    dense_n = max(top_k * 4, HYBRID_DENSE_CANDIDATES)
    sparse_n = max(top_k * 6, HYBRID_SPARSE_CANDIDATES)

    dense_kwargs = {
        "query_vec": query_vec,
        "filtro_area": filtro_area,
        "filtro_assunto": filtro_assunto,
        "qclient": qclient,
        "top_n": dense_n,
    }
    if "target_collection" in inspect.signature(_dense_ranked_candidates).parameters:
        dense_kwargs["target_collection"] = collection
    dense_ranked, dense_scores = _dense_ranked_candidates(**dense_kwargs)
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

def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    target_collection: str | list[str] = COLLECTION_NAME,
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
        
    top_k = max(1, int(top_k))

    collections = target_collection if isinstance(target_collection, list) else [target_collection]
    
    all_results = []
    for col in collections:
        all_results.extend(_search_single_collection(
            query=query, 
            top_k=top_k, 
            filtro_area=filtro_area, 
            filtro_assunto=filtro_assunto, 
            collection=col
        ))
        
    if len(collections) > 1:
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:top_k]
        
    return all_results

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


def build_context(results: list[dict]) -> str:
    return _build_context(results)


def _collection_for_retrieval_domain(retrieval_domain: str) -> str:
    try:
        return _COLLECTION_BY_RETRIEVAL_DOMAIN[retrieval_domain]
    except KeyError as exc:
        allowed = ", ".join(sorted(_COLLECTION_BY_RETRIEVAL_DOMAIN))
        raise ValueError(f"Dominio de recuperacao invalido: {retrieval_domain}. Permitidos: {allowed}") from exc


def search_domain(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    retrieval_domain: str = RETRIEVAL_DOMAIN_DOCUMENTS,
) -> list[dict]:
    return search(
        query=query,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        target_collection=_collection_for_retrieval_domain(retrieval_domain),
    )

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

if __name__ == "__main__":
    pass

