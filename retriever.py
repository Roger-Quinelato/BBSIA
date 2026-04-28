from __future__ import annotations
import argparse, ipaddress, json, math, os, re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, AsyncGenerator, Any
import httpx
import threading
import numpy as np
import requests
from config import get_env_bool, get_env_int, get_env_list, get_env_str
from vector_store import dense_ranked_candidates, get_local_qdrant_client, vector_store_health

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

INDEX_FILE = "index.faiss"

METADATA_FILE = "metadata.json"

MANIFEST_FILE = "manifest.json"

DEFAULT_TOP_K = get_env_int("TOP_K", 5, min_value=1, max_value=20)

GENERAL_AREAS = ["tecnologia", "ia", "saude", "infraestrutura", "juridico"]

MAX_CHARS_PER_CHUNK = get_env_int("MAX_CHARS_PER_CHUNK", 700, min_value=200, max_value=4000)

MIN_DENSE_SCORE_FOR_ANSWER = get_env_int("MIN_DENSE_SCORE_PERCENT", 18, min_value=0, max_value=100) / 100.0

HYBRID_DENSE_CANDIDATES = get_env_int("HYBRID_DENSE_CANDIDATES", 40, min_value=5, max_value=2000)

HYBRID_SPARSE_CANDIDATES = get_env_int("HYBRID_SPARSE_CANDIDATES", 80, min_value=5, max_value=5000)

RRF_K = get_env_int("RRF_K", 60, min_value=1, max_value=200)

PRELOAD_RAG_ON_STARTUP = get_env_bool("PRELOAD_RAG_ON_STARTUP", True)
PRELOAD_RERANKER_ON_STARTUP = get_env_bool("PRELOAD_RERANKER_ON_STARTUP", False)

HF_LOCAL_FILES_ONLY = get_env_bool("HF_LOCAL_FILES_ONLY", True)
E5_QUERY_PREFIX = "query: "

RERANKER_MODEL = get_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_CANDIDATES = get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200)

class IndexStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = None
        self._status = {
            "loaded_at_utc": None,
            "last_error": None,
            "last_preload_at_utc": None,
            "last_preload_error": None,
        }

    def get(self) -> dict:
        with self._lock:
            if self._data is None:
                self._data = self._load_from_disk()
                self._status["loaded_at_utc"] = self._data.get("loaded_at_utc")
                self._status["last_error"] = None
            return self._data

    def reload(self) -> None:
        new_data = self._load_from_disk()
        with self._lock:
            self._data = new_data
            self._status["loaded_at_utc"] = new_data.get("loaded_at_utc")
            self._status["last_error"] = None

    def get_status(self, key: str):
        with self._lock:
            return self._status.get(key)
            
    def set_status(self, key: str, value):
        with self._lock:
            self._status[key] = value

    def has_data(self) -> bool:
        with self._lock:
            return self._data is not None

    def get_data_if_loaded(self) -> dict | None:
        with self._lock:
            return self._data

    def _load_from_disk(self) -> dict:
        import json
        from sentence_transformers import SentenceTransformer
        
        base_dir = _script_dir()
        metadata_path = os.path.join(base_dir, INDEX_DIR, METADATA_FILE)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

        qclient = get_local_qdrant_client(DATA_DIR)

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
            "token_counts": token_counts,
            "doc_lengths": doc_lengths,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "reranker": None,
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

index_store = IndexStore()

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

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
    return index_store.get()

def reload_resources() -> None:
    index_store.reload()

def preload_resources(load_reranker: bool = PRELOAD_RERANKER_ON_STARTUP) -> dict:
    """Carrega indice, modelo de embeddings e, opcionalmente, re-ranker no cache."""
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        data = _load_resources()
        reranker_loaded = False
        if load_reranker and ENABLE_RERANKER:
            reranker_loaded = _get_reranker() is not None
        index_store.set_status("last_preload_at_utc", started_at)
        index_store.set_status("last_preload_error", None)
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
        index_store.set_status("last_error", message)
        index_store.set_status("last_preload_at_utc", started_at)
        index_store.set_status("last_preload_error", message)
        raise

def cache_health(load_if_empty: bool = False) -> dict:
    """Retorna um snapshot leve do cache/modelos sem obrigatoriamente carregar recursos."""
    if load_if_empty and not index_store.has_data():
        _load_resources()

    data = index_store.get_data_if_loaded() or {}
    chunks = data.get("chunks") or []
    embeddings = data.get("embeddings")
    qclient = data.get("qclient")
    embedding_count = len(chunks) if chunks else 0
    embedding_dim = EXPECTED_EMBEDDING_DIM

    return {
        "resources_cached": index_store.has_data(),
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
        "loaded_at_utc": data.get("loaded_at_utc") or index_store.get_status("loaded_at_utc"),
        "last_error": index_store.get_status("last_error"),
        "last_preload_at_utc": index_store.get_status("last_preload_at_utc"),
        "last_preload_error": index_store.get_status("last_preload_error"),
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
) -> tuple[list[int], dict[int, float]]:
    return dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=top_n,
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

from reranker import ENABLE_RERANKER, _get_reranker

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
    qclient = data["qclient"]
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
    if int(query_vec.shape[0]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Dimensao da consulta incompativel. "
            f"Consulta={int(query_vec.shape[0])}; esperado={EXPECTED_EMBEDDING_DIM}. "
        )

    dense_n = max(top_k * 4, HYBRID_DENSE_CANDIDATES)
    sparse_n = max(top_k * 6, HYBRID_SPARSE_CANDIDATES)

    dense_ranked, dense_scores = _dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
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

