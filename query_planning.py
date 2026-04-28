from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    query_text: str
    filtro_area: str | None = None
    filtro_assunto: str | None = None
    ano: int | None = None
    tipo_documento: str | None = None
    confidence: float = 0.0


_AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "infraestrutura": (
        "infraestrutura",
        "infra",
        "kubernetes",
        "cluster",
        "deploy",
        "devops",
        "qdrant",
        "docker",
    ),
    "juridico": (
        "juridico",
        "juridica",
        "lgpd",
        "lei",
        "privacidade",
        "conformidade",
        "regulacao",
        "etica",
    ),
    "saude": (
        "saude",
        "clinico",
        "clinica",
        "hospital",
        "paciente",
        "diagnostico",
        "medico",
    ),
    "ia": (
        "ia",
        "inteligencia artificial",
        "llm",
        "rag",
        "embedding",
        "embeddings",
        "modelo",
        "chatbot",
    ),
    "tecnologia": (
        "tecnologia",
        "software",
        "sistema",
        "api",
        "dados",
        "plataforma",
    ),
}

_ASSUNTO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "kubernetes": ("kubernetes", "k8s", "cluster"),
    "lgpd": ("lgpd", "privacidade", "dados pessoais"),
    "rag": ("rag", "retrieval augmented generation", "recuperacao aumentada"),
    "chatbot": ("chatbot", "assistente", "bot"),
    "embedding": ("embedding", "embeddings", "vetor", "vetorial"),
    "seguranca": ("seguranca", "seguro", "vulnerabilidade"),
    "conformidade": ("conformidade", "compliance", "regulacao"),
}

_TIPO_DOCUMENTO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "artigo_cientifico": ("artigo", "paper", "estudo"),
    "relatorio_tecnico": ("relatorio", "relatorio tecnico"),
    "manual": ("manual", "guia", "tutorial"),
    "apresentacao": ("apresentacao", "slides", "deck"),
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _best_match(text: str, candidates: dict[str, tuple[str, ...]]) -> tuple[str | None, int]:
    scores: dict[str, int] = {}
    for value, keywords in candidates.items():
        scores[value] = sum(1 for keyword in keywords if _contains_keyword(text, keyword))

    matched = [(value, score) for value, score in scores.items() if score > 0]
    if not matched:
        return None, 0
    matched.sort(key=lambda item: (-item[1], item[0]))
    return matched[0]


def _infer_year(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return None
    return int(match.group(0))


def plan_query(pergunta: str) -> QueryPlan:
    query_text = (pergunta or "").strip()
    normalized = _normalize(query_text)

    filtro_area, area_score = _best_match(normalized, _AREA_KEYWORDS)
    filtro_assunto, assunto_score = _best_match(normalized, _ASSUNTO_KEYWORDS)
    tipo_documento, tipo_score = _best_match(normalized, _TIPO_DOCUMENTO_KEYWORDS)
    ano = _infer_year(normalized)

    signals = area_score + assunto_score + tipo_score + (1 if ano is not None else 0)
    confidence = min(0.95, round(0.25 + signals * 0.15, 2)) if signals else 0.0

    return QueryPlan(
        query_text=query_text,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        ano=ano,
        tipo_documento=tipo_documento,
        confidence=confidence,
    )
