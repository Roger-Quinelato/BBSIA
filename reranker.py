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
from sentence_transformers import CrossEncoder

ENABLE_RERANKER = get_env_bool("ENABLE_RERANKER", True)

PRELOAD_RERANKER_ON_STARTUP = get_env_bool("PRELOAD_RERANKER_ON_STARTUP", False)

RERANKER_MODEL = get_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

RERANKER_CANDIDATES = get_env_int("RERANKER_CANDIDATES", 20, min_value=3, max_value=200)

RERANKER_TOP_N = get_env_int("RERANKER_TOP_N", 3, min_value=1, max_value=10)

RERANKER_MAX_LENGTH = get_env_int("RERANKER_MAX_LENGTH", 512, min_value=128, max_value=2048)

def _get_reranker() -> CrossEncoder | None:
    if not ENABLE_RERANKER:
        return None

    from retriever import _load_resources
    data = _load_resources()
    reranker = data.get("reranker")
    if reranker is None:
        try:
            reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=RERANKER_MAX_LENGTH,
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Re-ranker local '{RERANKER_MODEL}' nao esta disponivel. "
                "Baixe/cache o modelo no ambiente air-gapped ou defina ENABLE_RERANKER=false."
            ) from exc
        data["reranker"] = reranker
    return reranker

