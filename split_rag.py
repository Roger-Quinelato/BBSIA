import ast
import os

def split_rag():
    with open('rag_engine.py', 'r', encoding='utf-8') as f:
        source = f.read()
    
    lines = source.split('\n')
    
    tree = ast.parse(source)
    
    nodes = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if name is None and isinstance(node, ast.Assign):
            targets = [getattr(t, 'id', None) for t in node.targets]
            name = targets[0] if targets else 'Unknown'
        nodes.append({
            'name': name,
            'start': node.lineno - 1,
            'end': getattr(node, 'end_lineno', node.lineno) - 1,
            'type': type(node).__name__,
            'node': node
        })
    
    mapping = {
        'generator': [
            'OLLAMA_URL', 'DEFAULT_LLM_MODEL', 'ALLOWED_LLM_MODELS', 'ALLOW_REMOTE_OLLAMA',
            'OLLAMA_TIMEOUT_SEC', 'OLLAMA_NUM_PREDICT', 'OLLAMA_NUM_CTX', 'E5_QUERY_PREFIX',
            'SYSTEM_PROMPT', 'NO_EVIDENCE_RESPONSE', '_is_loopback_host', 'validate_ollama_endpoint',
            'validate_ollama_model', 'list_ollama_models', 'query_ollama', 'query_ollama_stream',
            'build_prompt'
        ],
        'reranker': [
            'ENABLE_RERANKER', 'RERANKER_MODEL', 'RERANKER_CANDIDATES', 'RERANKER_TOP_N',
            'RERANKER_MAX_LENGTH', 'PRELOAD_RERANKER_ON_STARTUP', '_get_reranker', '_rerank_results'
        ],
        'faithfulness': [
            '_declares_not_found', '_citation_labels', '_faithfulness_check',
            '_unique_sources'
        ],
        'pipeline': [
            'MAX_CONTEXT_CHUNKS', 'answer_question', 'answer_question_stream',
            '_extractive_grounded_answer', '_extractive_fallback_answer', '_retrieval_has_answer_signal'
        ],
    }
    
    files = {
        'generator': [],
        'reranker': [],
        'faithfulness': [],
        'pipeline': [],
        'retriever': []
    }
    
    for n in nodes:
        if n['type'] in ('Import', 'ImportFrom'):
            continue
            
        if n['name'] == 'main':
            continue
            
        target = 'retriever'
        for k, v in mapping.items():
            if n['name'] in v:
                target = k
                break
        
        start = n['start']
        while start > 0 and (lines[start-1].strip().startswith('#') or lines[start-1].strip().startswith('@')):
            start -= 1
        
        block = '\n'.join(lines[start:n['end']+1])
        files[target].append(block)

    base_imports = '\n'.join([
        "from __future__ import annotations",
        "import argparse, hashlib, ipaddress, json, math, os, re",
        "from urllib.parse import urlparse",
        "from collections import Counter, defaultdict",
        "from datetime import datetime, timezone",
        "from typing import Iterable, AsyncGenerator, Any",
        "import httpx",
        "import threading",
        "import numpy as np",
        "import requests",
        "from config import get_env_bool, get_env_int, get_env_list, get_env_str"
    ])
    
    # GENERATOR
    gen_content = base_imports + "\n\n" + '\n\n'.join(files['generator'])
    with open('generator.py', 'w', encoding='utf-8') as f: f.write(gen_content)
        
    # RERANKER
    reranker_content = base_imports + "\nfrom sentence_transformers import CrossEncoder\n\n"
    for b in files['reranker']:
        if "def _get_reranker(" in b:
            b = b.replace("data = _load_resources()", "from retriever import _load_resources\n    data = _load_resources()")
            b = b.replace("local_files_only=HF_LOCAL_FILES_ONLY", "local_files_only=False") # Fix local files constant
        reranker_content += b + "\n\n"
    with open('reranker.py', 'w', encoding='utf-8') as f: f.write(reranker_content)
        
    # FAITHFULNESS
    faith_content = base_imports + "\n\n"
    for b in files['faithfulness']:
        if "def _unique_sources" in b:
            b = b.replace("def _unique_sources", "from retriever import _format_source_label\n\ndef _unique_sources")
        if "def _faithfulness_check" in b:
            b = b.replace("def _faithfulness_check", "from retriever import _format_citation_label\n\ndef _faithfulness_check")
        faith_content += b + "\n\n"
    with open('faithfulness.py', 'w', encoding='utf-8') as f: f.write(faith_content)
        
    # RETRIEVER
    retriever_content = base_imports + "\nimport faiss\nfrom sentence_transformers import SentenceTransformer\n\n"
    for b in files['retriever']:
        if "def search(" in b:
            b = b.replace("def search(", "from reranker import _rerank_results, ENABLE_RERANKER\n\ndef search(")
        retriever_content += b + "\n\n"
    with open('retriever.py', 'w', encoding='utf-8') as f: f.write(retriever_content)

    # PIPELINE
    pipe_content = base_imports + "\n\n"
    for b in files['pipeline']:
        if "def answer_question(" in b:
            b = "from retriever import search, _build_context, MIN_DENSE_SCORE_FOR_ANSWER, DEFAULT_TOP_K\n" + \
                "from generator import query_ollama, build_prompt, DEFAULT_LLM_MODEL, NO_EVIDENCE_RESPONSE\n" + \
                "from faithfulness import _faithfulness_check, _unique_sources\n" + \
                "from reranker import RERANKER_TOP_N\n" + b
        if "async def answer_question_stream(" in b:
            b = b.replace("async def answer_question_stream(", "from generator import query_ollama_stream\n\nasync def answer_question_stream(")
        pipe_content += b + "\n\n"
    with open('pipeline.py', 'w', encoding='utf-8') as f: f.write(pipe_content)

    facade = '\"\"\"Facade motor RAG\"\"\"\n'
    facade += "from retriever import *\n"
    facade += "from reranker import *\n"
    facade += "from generator import *\n"
    facade += "from faithfulness import *\n"
    facade += "from pipeline import *\n"
    
    with open('rag_engine.py', 'w', encoding='utf-8') as f: f.write(facade)

split_rag()
