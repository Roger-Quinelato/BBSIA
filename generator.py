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

OLLAMA_URL = get_env_str("OLLAMA_URL", "http://localhost:11434")

DEFAULT_LLM_MODEL = get_env_str("DEFAULT_MODEL", "qwen3.5:7b-instruct")

ALLOWED_LLM_MODELS = set(get_env_list("ALLOWED_LLM_MODELS", [DEFAULT_LLM_MODEL]))

ALLOW_REMOTE_OLLAMA = get_env_bool("ALLOW_REMOTE_OLLAMA", False)

OLLAMA_TIMEOUT_SEC = get_env_int("OLLAMA_TIMEOUT_SEC", 300, min_value=30, max_value=3600)

OLLAMA_NUM_PREDICT = get_env_int("OLLAMA_NUM_PREDICT", 512, min_value=32, max_value=1024)

OLLAMA_NUM_CTX = get_env_int("OLLAMA_NUM_CTX", 8192, min_value=512, max_value=131072)

E5_QUERY_PREFIX = "query: "

NO_EVIDENCE_RESPONSE = "Não encontrei evidências suficientes nos documentos indexados."

SYSTEM_PROMPT = (
    "Você é o assistente técnico do BBSIA. Responda EXCLUSIVAMENTE com base no contexto "
    "fornecido. Se a resposta não estiver no contexto, responda: 'Não encontrei evidências "
    "suficientes nos documentos indexados.' Para cada afirmação, adicione a citação no formato "
    "(Sobrenome, Ano)."
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

def build_prompt(pergunta: str, context: str, history: list[dict[str, str]] | None = None) -> str:
    history_text = ""
    if history:
        # Pega as ultimas 3 interacoes para contexto
        recent = history[-6:]
        lines = []
        for msg in recent:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            lines.append(f"{role}: {msg['content']}")
        history_text = "--- HISTORICO DE CONVERSA ---\n" + "\n".join(lines) + "\n\n"

    return (
        f"{SYSTEM_PROMPT}\n\n"
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