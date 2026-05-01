"""Facade motor RAG"""

from bbsia.rag.orchestration.pipeline import (
    answer_question,
    answer_question_stream,
)
from bbsia.rag.generation.generator import (
    list_ollama_models,
    validate_ollama_endpoint,
    validate_ollama_model,
)
from bbsia.rag.retrieval.retriever import (
    PRELOAD_RAG_ON_STARTUP,
    PRELOAD_RERANKER_ON_STARTUP,
    cache_health,
    list_available_areas,
    list_available_assuntos,
    preload_resources,
    reload_resources,
    search,
)

__all__ = [
    "answer_question",
    "answer_question_stream",
    "cache_health",
    "list_available_areas",
    "list_available_assuntos",
    "list_ollama_models",
    "PRELOAD_RAG_ON_STARTUP",
    "PRELOAD_RERANKER_ON_STARTUP",
    "preload_resources",
    "reload_resources",
    "search",
    "validate_ollama_endpoint",
    "validate_ollama_model",
]
