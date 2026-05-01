from __future__ import annotations
import argparse, hashlib, ipaddress, json, math, os, re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, AsyncGenerator, Any
import unicodedata
import httpx
import threading
import numpy as np
import requests
from bbsia.core.config import get_env_bool, get_env_int, get_env_list, get_env_str

MAX_CONTEXT_CHUNKS = get_env_int("MAX_CONTEXT_CHUNKS", 6, min_value=1, max_value=10)
ENABLE_STREAM_FAITHFULNESS = get_env_bool("ENABLE_STREAM_FAITHFULNESS", False)
ENABLE_SYNC_FAITHFULNESS = get_env_bool("ENABLE_SYNC_FAITHFULNESS", False)

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

    prefix = "Com base nos trechos recuperados, a resposta mais segura e:"
    if reason:
        prefix += f"\n\nObservacao de fidelidade: {reason}."

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
        "Com base nos trechos recuperados, a resposta mais segura e:\n\n"
        + "\n".join(highlights)
        + f"\n\nPergunta original: {pergunta}\n"
        + f"Observacao tecnica: a geracao pelo LLM local falhou ({error})."
    )

from bbsia.rag.retrieval.retriever import search, _build_context, _format_citation_label, MIN_DENSE_SCORE_FOR_ANSWER, DEFAULT_TOP_K
from bbsia.infrastructure.vector_store import COLLECTION_NAME, COLLECTION_SOLUTIONS
from bbsia.rag.generation.generator import query_ollama, build_prompt, DEFAULT_LLM_MODEL, NO_EVIDENCE_RESPONSE, NO_SOLUTION_RESPONSE
from bbsia.rag.generation.faithfulness import _faithfulness_check, _unique_sources
from bbsia.rag.retrieval.reranker import RERANKER_TOP_N

DIAGNOSTIC_INTENT_TERMS = (
    "problema",
    "problemas",
    "sintoma",
    "sintomas",
    "causa",
    "causa raiz",
    "diagnostico",
    "diagnosticar",
    "resolver",
    "resolucao",
    "solucao",
    "solucoes",
    "recomendar",
    "recomendacao",
    "falha",
    "falhas",
    "erro",
    "erros",
    "lentidao",
    "demora",
    "gargalo",
    "backlog",
    "risco",
    "riscos",
    "priorizar",
)


def _normalize_intent_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents.lower()).strip()


def _is_diagnostic_query(pergunta: str) -> bool:
    normalized = _normalize_intent_text(pergunta)
    return any(term in normalized for term in DIAGNOSTIC_INTENT_TERMS)


def _tag_results(results: list[dict], retrieval_domain: str) -> list[dict]:
    tagged: list[dict] = []
    for item in results:
        enriched = dict(item)
        enriched["retrieval_domain"] = retrieval_domain
        tagged.append(enriched)
    return tagged


def _build_diagnostic_context(solution_results: list[dict], document_results: list[dict]) -> str:
    solution_context = _build_context(solution_results) if solution_results else "Nenhuma solucao candidata encontrada."
    document_context = _build_context(document_results) if document_results else "Nenhuma evidencia documental encontrada."
    return (
        "--- SOLUCOES CANDIDATAS ---\n"
        f"{solution_context}\n\n"
        "--- EVIDENCIAS DOCUMENTAIS ---\n"
        f"{document_context}"
    )


def _retrieve_for_answer(
    pergunta: str,
    top_k: int,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
) -> tuple[list[dict], list[dict], str | None, bool]:
    if not _is_diagnostic_query(pergunta):
        results = search(
            query=pergunta,
            top_k=top_k,
            filtro_area=filtro_area,
            filtro_assunto=filtro_assunto,
        )
        context_results = results[: min(MAX_CONTEXT_CHUNKS, RERANKER_TOP_N)]
        context = _build_context(context_results) if context_results else None
        return results, context_results, context, False

    solution_top_k = max(1, min(top_k, MAX_CONTEXT_CHUNKS))
    document_top_k = max(1, top_k)
    solution_results = _tag_results(
        search(
            query=pergunta,
            top_k=solution_top_k,
            filtro_area=filtro_area,
            filtro_assunto=filtro_assunto,
            target_collection=COLLECTION_SOLUTIONS,
        ),
        "solucoes",
    )
    document_results = _tag_results(
        search(
            query=pergunta,
            top_k=document_top_k,
            filtro_area=filtro_area,
            filtro_assunto=filtro_assunto,
            target_collection=COLLECTION_NAME,
        ),
        "documentos",
    )

    solution_context_results = solution_results[: min(MAX_CONTEXT_CHUNKS, RERANKER_TOP_N)]
    document_context_results = document_results[: min(MAX_CONTEXT_CHUNKS, RERANKER_TOP_N)]
    combined_results = solution_results + document_results
    context_results = solution_context_results + document_context_results
    context = _build_diagnostic_context(solution_context_results, document_context_results)
    return combined_results, context_results, context, True


def _has_catalog_solution(results: list[dict]) -> bool:
    return any(item.get("retrieval_domain") == "solucoes" for item in results)


def _llm_declined_with_available_context(resposta: str) -> bool:
    normalized = re.sub(r"\s+", " ", (resposta or "").strip().lower())
    return "não encontrei evidências suficientes" in normalized or "nao encontrei evidencias suficientes" in normalized

def answer_question(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict:
    results, context_results, context, is_diagnostic = _retrieve_for_answer(
        pergunta=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    if is_diagnostic and not _has_catalog_solution(results):
        return {
            "resposta": NO_SOLUTION_RESPONSE,
            "fontes": _unique_sources(results),
            "resultados": results,
            "prompt": None,
            "diagnostic_mode": True,
        }

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

    if context is None:
        context = _build_context(context_results)
    prompt = build_prompt(pergunta=pergunta, context=context, history=history, diagnostic_mode=is_diagnostic)
    try:
        resposta = query_ollama(prompt=prompt, model=model)
        if _llm_declined_with_available_context(resposta):
            resposta = _extractive_grounded_answer(pergunta=pergunta, results=context_results)
        if ENABLE_SYNC_FAITHFULNESS:
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
        "diagnostic_mode": is_diagnostic,
    }

from bbsia.rag.generation.generator import query_ollama_stream

async def answer_question_stream(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[dict, None]:
    results, context_results, context, is_diagnostic = _retrieve_for_answer(
        pergunta=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    fontes = _unique_sources(results)

    if is_diagnostic and not _has_catalog_solution(results):
        yield {
            "type": "metadata",
            "fontes": fontes,
            "resultados": results,
            "prompt": None,
            "diagnostic_mode": True,
        }
        yield {"type": "token", "token": NO_SOLUTION_RESPONSE}
        return

    if not results or not _retrieval_has_answer_signal(results):
        yield {
            "type": "metadata",
            "fontes": fontes,
            "resultados": results,
            "prompt": None,
            "diagnostic_mode": is_diagnostic,
        }
        yield {"type": "token", "token": NO_EVIDENCE_RESPONSE}
        return

    if context is None:
        context = _build_context(context_results)
    prompt = build_prompt(pergunta=pergunta, context=context, history=history, diagnostic_mode=is_diagnostic)

    yield {
        "type": "metadata",
        "fontes": fontes,
        "resultados": results,
        "prompt": prompt,
        "diagnostic_mode": is_diagnostic,
    }

    try:
        streamed_tokens: list[str] = []
        async for token in query_ollama_stream(prompt=prompt, model=model):
            streamed_tokens.append(token)
            yield {"type": "token", "token": token}

        if ENABLE_STREAM_FAITHFULNESS:
            resposta_final = "".join(streamed_tokens)
            faithful, reason = _faithfulness_check(resposta_final, context_results)
            event: dict[str, Any] = {
                "type": "faithfulness",
                "faithfulness_checked": True,
                "faithful": bool(faithful),
                "reason": reason,
            }
            if not faithful:
                event["fallback_response"] = _extractive_grounded_answer(
                    pergunta=pergunta,
                    results=context_results,
                    reason=reason,
                )
            yield event
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}

