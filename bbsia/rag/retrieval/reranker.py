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
from sentence_transformers import CrossEncoder

ENABLE_RERANKER = settings.reranker.enable_reranker

PRELOAD_RERANKER_ON_STARTUP = settings.reranker.preload_reranker_on_startup

RERANKER_MODEL = settings.reranker.reranker_model

RERANKER_CANDIDATES = settings.reranker.reranker_candidates

RERANKER_TOP_N = settings.reranker.reranker_top_n

RERANKER_MAX_LENGTH = settings.reranker.reranker_max_length

def _get_reranker() -> CrossEncoder | None:
    if not ENABLE_RERANKER:
        return None

    from bbsia.rag.retrieval.retriever import _load_resources
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

