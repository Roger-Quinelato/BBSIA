import os
import re

def update_rag_engine():
    path = "rag_engine.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Imports
    content = content.replace("from typing import Iterable", "from typing import Iterable, AsyncGenerator\nimport httpx\nimport threading")

    # Variables
    content = content.replace("MAX_CONTEXT_CHUNKS = get_env_int(\"MAX_CONTEXT_CHUNKS\", 3", "MAX_CONTEXT_CHUNKS = get_env_int(\"MAX_CONTEXT_CHUNKS\", 6")
    content = content.replace("OLLAMA_NUM_PREDICT = get_env_int(\"OLLAMA_NUM_PREDICT\", 120, min_value=32, max_value=512)", "OLLAMA_NUM_PREDICT = get_env_int(\"OLLAMA_NUM_PREDICT\", 512, min_value=32, max_value=1024)")
    content = content.replace("RERANKER_MODEL = get_env_str(\"RERANKER_MODEL\", \"BAAI/bge-reranker-v2-m3\")", "RERANKER_MODEL = get_env_str(\"RERANKER_MODEL\", \"cross-encoder/ms-marco-MiniLM-L-6-v2\")")

    # IndexStore
    cache_str = """_CACHE: dict = {}
_CACHE_STATUS: dict = {
    "loaded_at_utc": None,
    "last_error": None,
    "last_preload_at_utc": None,
    "last_preload_error": None,
}"""
    index_store_str = """class IndexStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = None
        self._status = {
            "loaded_at_utc": None,
            "last_error": None,
            "last_preload_at_utc": None,
            "last_preload_error": None,
        }

    def get(self) -> dict:
        with self._lock:
            if self._data is None:
                self._data = self._load_from_disk()
                self._status["loaded_at_utc"] = self._data.get("loaded_at_utc")
                self._status["last_error"] = None
            return self._data

    def reload(self) -> None:
        new_data = self._load_from_disk()
        with self._lock:
            self._data = new_data
            self._status["loaded_at_utc"] = new_data.get("loaded_at_utc")
            self._status["last_error"] = None

    def get_status(self, key: str):
        with self._lock:
            return self._status.get(key)
            
    def set_status(self, key: str, value):
        with self._lock:
            self._status[key] = value

    def has_data(self) -> bool:
        with self._lock:
            return self._data is not None

    def get_data_if_loaded(self) -> dict | None:
        with self._lock:
            return self._data

    def _load_from_disk(self) -> dict:
        import json
        import faiss
        from sentence_transformers import SentenceTransformer
        base_dir = _script_dir()
        index_path = os.path.join(base_dir, INDEX_DIR, INDEX_FILE)
        metadata_path = os.path.join(base_dir, INDEX_DIR, METADATA_FILE)
        manifest_path = os.path.join(base_dir, INDEX_DIR, MANIFEST_FILE)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Indice FAISS nao encontrado em: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

        _verify_index_manifest(index_path, metadata_path, manifest_path)
        index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if isinstance(metadata, dict) and "chunks" in metadata:
            chunks = metadata["chunks"]
            embedding_model = metadata.get("model_name", EMBEDDING_MODEL_FALLBACK)
        elif isinstance(metadata, list):
            chunks = metadata
            embedding_model = EMBEDDING_MODEL_FALLBACK
        else:
            raise ValueError("Formato de metadata invalido.")

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Metadata nao possui chunks validos.")

        try:
            model = SentenceTransformer(embedding_model, local_files_only=HF_LOCAL_FILES_ONLY)
        except Exception as exc:
            raise RuntimeError("Modelo indisponivel") from exc
            
        embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)
        token_counts, doc_lengths, doc_freq, avgdl = _build_sparse_index(chunks)

        return {
            "index": index,
            "chunks": chunks,
            "model": model,
            "embeddings": embeddings,
            "embedding_model": embedding_model,
            "token_counts": token_counts,
            "doc_lengths": doc_lengths,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "reranker": None,
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

index_store = IndexStore()"""
    content = content.replace(cache_str, index_store_str)

    # _load_resources
    old_load = """def _load_resources() -> dict:
    if _CACHE:
        return _CACHE

    base_dir = _script_dir()
    index_path = os.path.join(base_dir, INDEX_DIR, INDEX_FILE)
    metadata_path = os.path.join(base_dir, INDEX_DIR, METADATA_FILE)
    manifest_path = os.path.join(base_dir, INDEX_DIR, MANIFEST_FILE)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Indice FAISS nao encontrado em: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

    _verify_index_manifest(index_path, metadata_path, manifest_path)
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if isinstance(metadata, dict) and "chunks" in metadata:
        chunks = metadata["chunks"]
        embedding_model = metadata.get("model_name", EMBEDDING_MODEL_FALLBACK)
    elif isinstance(metadata, list):
        chunks = metadata
        embedding_model = EMBEDDING_MODEL_FALLBACK
    else:
        raise ValueError("Formato de metadata invalido.")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Metadata nao possui chunks validos.")

    metadata_dim = int(metadata.get("embedding_dim", 0)) if isinstance(metadata, dict) else 0
    index_dim = int(index.d)
    if metadata_dim and metadata_dim != index_dim:
        raise ValueError(
            "Metadata e indice FAISS estao inconsistentes: "
            f"metadata={metadata_dim}, indice={index_dim}. Execute /reprocessar."
        )
    if index_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Indice FAISS incompativel com o modelo de embeddings atual. "
            f"Indice={index_dim} dimensoes; esperado={EXPECTED_EMBEDDING_DIM}. "
            "Execute /reprocessar para recriar o indice com intfloat/multilingual-e5-large."
        )

    try:
        model = SentenceTransformer(embedding_model, local_files_only=HF_LOCAL_FILES_ONLY)
    except Exception as exc:
        raise RuntimeError(
            f"Modelo de embeddings local '{embedding_model}' nao esta disponivel. "
            "Pre-carregue o modelo no cache/local path ou defina HF_LOCAL_FILES_ONLY=false "
            "apenas durante preparacao do ambiente."
        ) from exc
    embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    token_counts, doc_lengths, doc_freq, avgdl = _build_sparse_index(chunks)

    _CACHE.update(
        {
            "index": index,
            "chunks": chunks,
            "model": model,
            "embeddings": embeddings,
            "embedding_model": embedding_model,
            "token_counts": token_counts,
            "doc_lengths": doc_lengths,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "reranker": None,
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _CACHE_STATUS["loaded_at_utc"] = _CACHE["loaded_at_utc"]
    _CACHE_STATUS["last_error"] = None
    return _CACHE"""
    new_load = """def _load_resources() -> dict:
    return index_store.get()"""
    content = content.replace(old_load, new_load)

    # reload
    content = content.replace("_CACHE.clear()\n    _CACHE_STATUS[\"loaded_at_utc\"] = None", "index_store.reload()")

    # preload
    content = content.replace("_CACHE_STATUS[\"last_preload_at_utc\"] = started_at", "index_store.set_status(\"last_preload_at_utc\", started_at)")
    content = content.replace("_CACHE_STATUS[\"last_preload_error\"] = None", "index_store.set_status(\"last_preload_error\", None)")
    content = content.replace("_CACHE_STATUS[\"last_error\"] = message", "index_store.set_status(\"last_error\", message)")
    content = content.replace("_CACHE_STATUS[\"last_preload_error\"] = message", "index_store.set_status(\"last_preload_error\", message)")

    # cache_health
    old_health = """    if load_if_empty and not _CACHE:
        _load_resources()

    chunks = _CACHE.get("chunks") or []
    index = _CACHE.get("index")
    embeddings = _CACHE.get("embeddings")"""
    new_health = """    if load_if_empty and not index_store.has_data():
        _load_resources()

    data = index_store.get_data_if_loaded() or {}
    chunks = data.get("chunks") or []
    index = data.get("index")
    embeddings = data.get("embeddings")"""
    content = content.replace(old_health, new_health)
    
    content = content.replace("bool(_CACHE)", "index_store.has_data()")
    content = content.replace("bool(_CACHE.get(\"model\"))", "bool(data.get(\"model\"))")
    content = content.replace("bool(_CACHE.get(\"reranker\"))", "bool(data.get(\"reranker\"))")
    content = content.replace("_CACHE.get(\"embedding_model\")", "data.get(\"embedding_model\")")
    content = content.replace("_CACHE.get(\"loaded_at_utc\") or _CACHE_STATUS.get(\"loaded_at_utc\")", "data.get(\"loaded_at_utc\") or index_store.get_status(\"loaded_at_utc\")")
    content = content.replace("_CACHE_STATUS.get(\"last_error\")", "index_store.get_status(\"last_error\")")
    content = content.replace("_CACHE_STATUS.get(\"last_preload_at_utc\")", "index_store.get_status(\"last_preload_at_utc\")")
    content = content.replace("_CACHE_STATUS.get(\"last_preload_error\")", "index_store.get_status(\"last_preload_error\")")

    # query_ollama_stream
    stream_fn = """
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
"""
    content = content.replace("def _unique_sources", stream_fn + "\n\ndef _unique_sources")
    
    answer_stream = """
async def answer_question_stream(
    pergunta: str,
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
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
    prompt = build_prompt(pergunta=pergunta, context=context)

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
"""
    content = content.replace("def main() -> None:", answer_stream + "\n\ndef main() -> None:")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

update_rag_engine()
