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
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import requests
from config import get_env_bool, get_env_int, get_env_str

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale faiss-cpu para usar rag_engine.py") from exc

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Instale sentence-transformers para usar rag_engine.py") from exc


EMBEDDING_MODEL_FALLBACK = get_env_str("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
OLLAMA_URL = get_env_str("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = get_env_str("DEFAULT_MODEL", "llama3.1:8b")
DEFAULT_TOP_K = get_env_int("TOP_K", 5, min_value=1, max_value=20)
GENERAL_AREAS = ["tecnologia", "ia", "saude", "infraestrutura", "juridico"]

MAX_CONTEXT_CHUNKS = get_env_int("MAX_CONTEXT_CHUNKS", 3, min_value=1, max_value=10)
MAX_CHARS_PER_CHUNK = get_env_int("MAX_CHARS_PER_CHUNK", 700, min_value=200, max_value=4000)
OLLAMA_TIMEOUT_SEC = get_env_int("OLLAMA_TIMEOUT_SEC", 300, min_value=30, max_value=3600)
OLLAMA_NUM_PREDICT = get_env_int("OLLAMA_NUM_PREDICT", 120, min_value=32, max_value=512)
OLLAMA_NUM_CTX = get_env_int("OLLAMA_NUM_CTX", 2048, min_value=512, max_value=32768)
MIN_DENSE_SCORE_FOR_ANSWER = get_env_int("MIN_DENSE_SCORE_PERCENT", 18, min_value=0, max_value=100) / 100.0

HYBRID_DENSE_CANDIDATES = get_env_int("HYBRID_DENSE_CANDIDATES", 40, min_value=5, max_value=2000)
HYBRID_SPARSE_CANDIDATES = get_env_int("HYBRID_SPARSE_CANDIDATES", 80, min_value=5, max_value=5000)
RRF_K = get_env_int("RRF_K", 60, min_value=1, max_value=200)

ENABLE_RERANKER = get_env_bool("ENABLE_RERANKER", False)
RERANKER_MODEL = get_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_CANDIDATES = get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200)

PROMPT_TEMPLATE = """Você é um assistente especializado no projeto BBSIA (Banco Brasileiro de Soluções de IA).

REGRAS:
- Responda apenas com base nos documentos recuperados.
- Se a informação não estiver claramente no contexto, responda exatamente que não encontrou suporte suficiente nos documentos.
- Toda afirmação factual deve ter citação inline no formato [Fonte N].
- Não use conhecimento externo, inferências livres ou suposições.
- Se houver conflito entre fontes, explicite o conflito com as citações.
- Responda em português brasileiro formal e objetivo.

--- DOCUMENTOS RECUPERADOS ---
{contexto}

--- PERGUNTA ---
{pergunta}

--- RESPOSTA ---
"""

_CACHE: dict = {}


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

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice FAISS nao encontrado em: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

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

    model = SentenceTransformer(embedding_model)
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
        }
    )
    return _CACHE


def _get_reranker() -> CrossEncoder | None:
    if not ENABLE_RERANKER:
        return None

    data = _load_resources()
    reranker = data.get("reranker")
    if reranker is None:
        reranker = CrossEncoder(RERANKER_MODEL)
        data["reranker"] = reranker
    return reranker


def reload_resources() -> None:
    _CACHE.clear()


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
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout_sec)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = sorted({m.get("name") for m in models if m.get("name")})
        return names or [DEFAULT_LLM_MODEL]
    except Exception:
        return [DEFAULT_LLM_MODEL]


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

    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

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


def _build_context(results: list[dict]) -> str:
    parts = []
    for i, item in enumerate(results, start=1):
        trecho = str(item.get("parent_text") or item.get("texto", ""))
        if len(trecho) > MAX_CHARS_PER_CHUNK:
            trecho = trecho[:MAX_CHARS_PER_CHUNK].rstrip() + "..."
        section = item.get("section_heading") or "nao informado"
        content_type = item.get("content_type") or "text"
        source_label = _format_source_label(item)
        parts.append(
            "\n".join(
                [
                    f"[Fonte {i}] {source_label} (p. {item.get('pagina', '?')})",
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
    return PROMPT_TEMPLATE.format(contexto=context, pergunta=pergunta)


def query_ollama(prompt: str, model: str = DEFAULT_LLM_MODEL, timeout_sec: int = OLLAMA_TIMEOUT_SEC) -> str:
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
        "nao encontrei",
        "não encontrei",
        "nao foi encontrado",
        "não foi encontrado",
        "suporte suficiente",
        "nao ha informacao",
        "não há informação",
    ]
    return any(marker in normalized for marker in markers)


def _citation_numbers(answer: str) -> set[int]:
    return {int(match) for match in re.findall(r"\[Fonte\s+(\d+)\]", answer or "", flags=re.IGNORECASE)}


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

    citations = _citation_numbers(answer)
    if not citations:
        return False, "resposta sem citacoes inline"

    valid_sources = set(range(1, len(context_results) + 1))
    invalid = sorted(citations - valid_sources)
    if invalid:
        return False, f"citacoes inexistentes: {invalid}"

    factual_lines = [
        line.strip()
        for line in re.split(r"[\n]+", answer)
        if len(line.strip().split()) >= 8 and not _declares_not_found(line)
    ]
    uncited_lines = [line for line in factual_lines if not _citation_numbers(line)]
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


def _extractive_grounded_answer(pergunta: str, results: list[dict], reason: str | None = None) -> str:
    highlights: list[str] = []
    for idx, item in enumerate(results[:MAX_CONTEXT_CHUNKS], start=1):
        texto = str(item.get("parent_text") or item.get("texto", "")).strip()
        if not texto:
            continue
        resumo = texto.replace("\n", " ").strip()
        if len(resumo) > 320:
            resumo = resumo[:320].rstrip() + "..."
        highlights.append(f"- {resumo} [Fonte {idx}]")

    if not highlights:
        return "Nao encontrei suporte suficiente nos documentos recuperados para responder com seguranca."

    prefix = "Nao encontrei suporte suficiente para gerar uma resposta sintetica com seguranca."
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
    for idx, item in enumerate(results[:3], start=1):
        texto = str(item.get("parent_text") or item.get("texto", "")).strip()
        if not texto:
            continue
        resumo = texto.replace("\n", " ").strip()
        if len(resumo) > 240:
            resumo = resumo[:240].rstrip() + "..."
        src = f"{item.get('documento', 'desconhecido')} (p. {item.get('pagina', '?')})"
        highlights.append(f"- {resumo} [Fonte {idx}: {src}]")

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
            "resposta": "Nao encontrei informacoes relevantes nos documentos para responder a pergunta.",
            "fontes": [],
            "resultados": [],
            "prompt": None,
        }

    if not _retrieval_has_answer_signal(results):
        return {
            "resposta": "Nao encontrei suporte suficiente nos documentos recuperados para responder com seguranca.",
            "fontes": _unique_sources(results),
            "resultados": results,
            "prompt": None,
        }

    context_results = results[:MAX_CONTEXT_CHUNKS]
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
