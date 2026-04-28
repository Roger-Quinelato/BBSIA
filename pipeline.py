from __future__ import annotations
import argparse, hashlib, ipaddress, json, math, os, re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, AsyncGenerator, Any
import httpx
import threading
import numpy as np
import requests
from config import get_env_bool, get_env_int, get_env_list, get_env_str

MAX_CONTEXT_CHUNKS = get_env_int("MAX_CONTEXT_CHUNKS", 6, min_value=1, max_value=10)

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

from retriever import search, _build_context, _format_citation_label, MIN_DENSE_SCORE_FOR_ANSWER, DEFAULT_TOP_K
from generator import query_ollama, build_prompt, DEFAULT_LLM_MODEL, NO_EVIDENCE_RESPONSE
from faithfulness import _faithfulness_check, _unique_sources
from reranker import RERANKER_TOP_N
def answer_question(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    history: list[dict[str, str]] | None = None,
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
    prompt = build_prompt(pergunta=pergunta, context=context, history=history)
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

from generator import query_ollama_stream

async def answer_question_stream(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[dict, None]:
    results = search(
        query=pergunta,
        top_k=top_k,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
    )

    fontes = _unique_sources(results)

    if not results or not _retrieval_has_answer_signal(results):
        yield {
            "type": "metadata",
            "fontes": fontes,
            "resultados": results,
            "prompt": None,
        }
        yield {"type": "token", "token": NO_EVIDENCE_RESPONSE}
        return

    context_limit = min(MAX_CONTEXT_CHUNKS, RERANKER_TOP_N)
    context_results = results[:context_limit]
    context = _build_context(context_results)
    prompt = build_prompt(pergunta=pergunta, context=context, history=history)

    yield {
        "type": "metadata",
        "fontes": fontes,
        "resultados": results,
        "prompt": prompt,
    }

    try:
        async for token in query_ollama_stream(prompt=prompt, model=model):
            yield {"type": "token", "token": token}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}

