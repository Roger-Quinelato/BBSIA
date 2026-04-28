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

from retriever import _format_source_label

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

ENABLE_NLI_FAITHFULNESS = get_env_bool("ENABLE_NLI_FAITHFULNESS", True)
NLI_MODEL_NAME = get_env_str("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

_nli_model = None
_nli_lock = threading.Lock()

def _get_nli_model():
    global _nli_model
    if not ENABLE_NLI_FAITHFULNESS:
        return None
    with _nli_lock:
        if _nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                _nli_model = CrossEncoder(NLI_MODEL_NAME, max_length=512)
            except Exception as exc:
                raise RuntimeError(
                    f"Modelo NLI '{NLI_MODEL_NAME}' indisponivel. "
                    "Baixe o modelo no ambiente air-gapped ou desative ENABLE_NLI_FAITHFULNESS."
                ) from exc
    return _nli_model

def _faithfulness_check(answer: str, context_results: list[dict]) -> tuple[bool, str | None]:
    """
    Checagem rigorosa de grounding usando NLI (Natural Language Inference).
    Garante que a resposta gerada nao contem alucinacoes em relacao ao contexto.
    """
    answer = (answer or "").strip()
    if not answer:
        return False, "resposta vazia"
    if _declares_not_found(answer):
        return True, None

    model = _get_nli_model()
    if not model:
        # Se NLI estiver desativado, falha para modo extração (ou ignora a checagem)
        # Por padrao, se NLI esta desativado e não temos outra checagem, consideramos true.
        return True, None

    # Extrai as sentenças factuais da resposta
    # Heuristica simples: divide por pontuação e ignora sentenças muito curtas
    factual_lines = [
        line.strip()
        for line in re.split(r'(?<=[.!?])\s+', answer)
        if len(line.strip().split()) >= 5 and not _declares_not_found(line)
    ]
    
    if not factual_lines:
        return True, None

    # Concatena o texto do contexto recuperado para formar a 'Premissa'
    context_text = " ".join([str(item.get("texto", "")).strip() for item in context_results])
    if not context_text:
        return False, "contexto vazio, nao pode suportar a resposta"

    # NLI avalia (Premissa, Hipotese)
    pairs = [(context_text, sentence) for sentence in factual_lines]
    scores = model.predict(pairs)

    # Para cross-encoder/nli-deberta-v3-small: 0=contradiction, 1=entailment, 2=neutral
    # Precisamos que a hipotese seja entailment (argmax == 1)
    for idx, logits in enumerate(scores):
        prediction = np.argmax(logits)
        if prediction != 1:  # Se for contradicao (0) ou neutro (2)
            label_str = "contradicao" if prediction == 0 else "sem suporte (neutro)"
            return False, f"sentenca nao suportada pelo contexto ({label_str}): '{factual_lines[idx]}'"

    return True, None

