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
from bbsia.core.config import settings

OLLAMA_URL = settings.ollama_generation.ollama_url

DEFAULT_LLM_MODEL = settings.ollama_generation.default_model

ALLOWED_LLM_MODELS = set(settings.ollama_generation.allowed_llm_models)

ALLOW_REMOTE_OLLAMA = settings.ollama_generation.allow_remote_ollama

OLLAMA_TIMEOUT_SEC = settings.ollama_generation.ollama_timeout_sec

OLLAMA_NUM_PREDICT = settings.ollama_generation.ollama_num_predict

OLLAMA_NUM_CTX = settings.ollama_generation.ollama_num_ctx

E5_QUERY_PREFIX = "query: "

NO_EVIDENCE_RESPONSE = "Não encontrei evidências suficientes nos documentos indexados."
NO_SOLUTION_RESPONSE = (
    "Nao encontrei uma solucao candidata no catalogo de solucoes piloto para esse problema. "
    "Posso resumir evidencias documentais recuperadas, mas nao vou recomendar passos de implantacao sem uma solucao catalogada."
)

SYSTEM_PROMPT = (
    "Você é o assistente técnico do BBSIA. Responda sempre em português e use somente o contexto fornecido. "
    "Quando o contexto trouxer trechos relevantes, sintetize a resposta em formato claro e estruturado, "
    "com 3 a 5 pontos objetivos quando isso ajudar. Se o contexto tratar de infraestrutura, modelos, RAG, "
    "LLMs, GPUs, nuvem, segurança ou governança relacionados ao BBSIA, considere-o relevante para perguntas "
    "sobre o chatbot mesmo que a palavra exata da pergunta não apareça. Não recomende buscar fora do sistema. "
    "Use a frase 'Não encontrei evidências suficientes nos documentos indexados.' apenas se o contexto estiver "
    "vazio, irrelevante ou contraditório para a pergunta. "
    "Inclua citações no formato (Sobrenome, Ano) nas afirmações baseadas nas fontes."
)

DIAGNOSTIC_CONTEXT_MARKER = "--- SOLUCOES CANDIDATAS ---"

DIAGNOSTIC_PROMPT = (
    "Modo diagnostico de problemas: use as SOLUCOES CANDIDATAS como fonte principal e as "
    "EVIDENCIAS DOCUMENTAIS apenas como apoio. Nao invente causa raiz, passos, riscos, "
    "pre-condicoes ou restricoes que nao estejam no contexto. Se nao houver solucao candidata "
    "do catalogo, responda somente com a mensagem conservadora de indisponibilidade de solucao. "
    "Quando houver solucao candidata, responda obrigatoriamente com estas secoes, nesta ordem: "
    "Diagnostico, Solucao Recomendada, Passos, Riscos. Em Solucao Recomendada, cite o nome da "
    "solucao catalogada. Em Passos e Riscos, use apenas itens presentes na solucao candidata."
)

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

def build_prompt(
    pergunta: str,
    context: str,
    history: list[dict[str, str]] | None = None,
    diagnostic_mode: bool | None = None,
) -> str:
    history_text = ""
    if history:
        # Pega as ultimas 3 interacoes para contexto
        recent = history[-6:]
        lines = []
        for msg in recent:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            lines.append(f"{role}: {msg['content']}")
        history_text = "--- HISTORICO DE CONVERSA ---\n" + "\n".join(lines) + "\n\n"

    is_diagnostic = diagnostic_mode if diagnostic_mode is not None else DIAGNOSTIC_CONTEXT_MARKER in (context or "")
    system_prompt = SYSTEM_PROMPT
    if is_diagnostic:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{DIAGNOSTIC_PROMPT}\nMensagem conservadora: {NO_SOLUTION_RESPONSE}"

    return (
        f"{system_prompt}\n\n"
        f"{history_text}"
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

async def query_ollama_stream(
    prompt: str,
    model: str = DEFAULT_LLM_MODEL,
    timeout_sec: int = OLLAMA_TIMEOUT_SEC
) -> AsyncGenerator[str, None]:
    validate_ollama_endpoint()
    model = validate_ollama_model(model)
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", f"{OLLAMA_URL}/api/generate",
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
            timeout=httpx.Timeout(10.0, read=float(timeout_sec)),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if piece := payload.get("response"):
                    yield piece
                if payload.get("done"):
                    break
