import os
import re

RETRIEVER_PATH = r"c:\Users\roger\OneDrive\Documentos\CIIA\BBSIA - Copia\retriever.py"

with open(RETRIEVER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
content = content.replace(
    "from vector_store import dense_ranked_candidates, get_local_qdrant_client, vector_store_health",
    "from vector_store import dense_ranked_candidates, get_local_qdrant_client, vector_store_health, COLLECTION_NAME, COLLECTION_SOLUTIONS"
)

# 2. Rewrite IndexStore class
old_index_store = """class IndexStore:
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

    def close(self) -> None:
        with self._lock:
            data = self._data
            self._data = None
        qclient = data.get("qclient") if isinstance(data, dict) else None
        if qclient is not None and hasattr(qclient, "close"):
            try:
                qclient.close()
            except Exception:
                pass

    def _load_from_disk(self) -> dict:
        import json
        from sentence_transformers import SentenceTransformer
        
        base_dir = _script_dir()
        metadata_path = _resolve_metadata_path(base_dir)"""

new_index_store = """class IndexStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}
        self._status = {}

    def _init_status(self, collection: str):
        if collection not in self._status:
            self._status[collection] = {
                "loaded_at_utc": None,
                "last_error": None,
                "last_preload_at_utc": None,
                "last_preload_error": None,
            }

    def get(self, collection: str = COLLECTION_NAME) -> dict:
        with self._lock:
            self._init_status(collection)
            if collection not in self._data:
                self._data[collection] = self._load_from_disk(collection)
                self._status[collection]["loaded_at_utc"] = self._data[collection].get("loaded_at_utc")
                self._status[collection]["last_error"] = None
            return self._data[collection]

    def reload(self, collection: str = COLLECTION_NAME) -> None:
        new_data = self._load_from_disk(collection)
        with self._lock:
            self._init_status(collection)
            self._data[collection] = new_data
            self._status[collection]["loaded_at_utc"] = new_data.get("loaded_at_utc")
            self._status[collection]["last_error"] = None

    def get_status(self, key: str, collection: str = COLLECTION_NAME):
        with self._lock:
            return self._status.get(collection, {}).get(key)
            
    def set_status(self, key: str, value, collection: str = COLLECTION_NAME):
        with self._lock:
            self._init_status(collection)
            self._status[collection][key] = value

    def has_data(self, collection: str = COLLECTION_NAME) -> bool:
        with self._lock:
            return collection in self._data

    def get_data_if_loaded(self, collection: str = COLLECTION_NAME) -> dict | None:
        with self._lock:
            return self._data.get(collection)

    def close(self) -> None:
        with self._lock:
            for col, data in self._data.items():
                qclient = data.get("qclient")
                if qclient is not None and hasattr(qclient, "close"):
                    try:
                        qclient.close()
                    except Exception:
                        pass
            self._data.clear()

    def _load_from_disk(self, collection: str) -> dict:
        import json
        from sentence_transformers import SentenceTransformer
        
        metadata_path = _resolve_metadata_path(collection)"""

content = content.replace(old_index_store, new_index_store)

# 3. Update _resolve_metadata_path
old_resolve = """def _resolve_metadata_path(base_dir: str) -> str:
    \"\"\"Usa metadata oficial nova e aceita fallback legado temporario.\"\"\"
    preferred = os.path.join(base_dir, METADATA_DIR, METADATA_FILE)
    legacy = os.path.join(base_dir, LEGACY_METADATA_DIR, METADATA_FILE)
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(f"Metadata nao encontrada em: {preferred} nem em {legacy}")"""

new_resolve = """def _resolve_metadata_path(collection: str) -> str:
    \"\"\"Usa metadata oficial nova e aceita fallback legado temporario.\"\"\"
    if collection == COLLECTION_SOLUTIONS:
        preferred = os.path.join(DATA_DIR, "solucoes_qdrant_metadata", METADATA_FILE)
    else:
        preferred = os.path.join(DATA_DIR, "qdrant_index_metadata", METADATA_FILE)
        
    legacy = os.path.join(DATA_DIR, "faiss_index", METADATA_FILE)
    if os.path.exists(preferred):
        return preferred
    if collection == COLLECTION_NAME and os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(f"Metadata nao encontrada para colecao {collection}")"""

content = content.replace(old_resolve, new_resolve)

# 4. Update function signatures for resources
content = content.replace("def _load_resources() -> dict:", "def _load_resources(collection: str = COLLECTION_NAME) -> dict:")
content = content.replace("return index_store.get()", "return index_store.get(collection)")

content = content.replace("def reload_resources() -> None:", "def reload_resources(collection: str = COLLECTION_NAME) -> None:")
content = content.replace("index_store.reload()", "index_store.reload(collection)")

content = content.replace("def preload_resources(load_reranker: bool = PRELOAD_RERANKER_ON_STARTUP) -> dict:", "def preload_resources(load_reranker: bool = PRELOAD_RERANKER_ON_STARTUP, collection: str = COLLECTION_NAME) -> dict:")
content = content.replace("data = _load_resources()", "data = _load_resources(collection)")
content = content.replace('index_store.set_status("last_preload_at_utc", started_at)', 'index_store.set_status("last_preload_at_utc", started_at, collection)')
content = content.replace('index_store.set_status("last_preload_error", None)', 'index_store.set_status("last_preload_error", None, collection)')
content = content.replace('index_store.set_status("last_error", message)', 'index_store.set_status("last_error", message, collection)')
content = content.replace('index_store.set_status("last_preload_error", message)', 'index_store.set_status("last_preload_error", message, collection)')

content = content.replace("def cache_health(load_if_empty: bool = False) -> dict:", "def cache_health(load_if_empty: bool = False, collection: str = COLLECTION_NAME) -> dict:")
content = content.replace("if load_if_empty and not index_store.has_data():", "if load_if_empty and not index_store.has_data(collection):")
content = content.replace("_load_resources()", "_load_resources(collection)")
content = content.replace("index_store.get_data_if_loaded()", "index_store.get_data_if_loaded(collection)")
content = content.replace("index_store.has_data()", "index_store.has_data(collection)")
content = content.replace('index_store.get_status("loaded_at_utc")', 'index_store.get_status("loaded_at_utc", collection)')
content = content.replace('index_store.get_status("last_error")', 'index_store.get_status("last_error", collection)')
content = content.replace('index_store.get_status("last_preload_at_utc")', 'index_store.get_status("last_preload_at_utc", collection)')
content = content.replace('index_store.get_status("last_preload_error")', 'index_store.get_status("last_preload_error", collection)')

content = content.replace("def list_available_areas() -> list[str]:", "def list_available_areas(collection: str = COLLECTION_NAME) -> list[str]:")
content = content.replace("def list_available_assuntos() -> list[str]:", "def list_available_assuntos(collection: str = COLLECTION_NAME) -> list[str]:")

# 5. _dense_ranked_candidates
old_dense_ranked = """def _dense_ranked_candidates(
    query_vec: np.ndarray,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    qclient: Any,
    top_n: int,
) -> tuple[list[int], dict[int, float]]:
    return dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=top_n,
    )"""

new_dense_ranked = """def _dense_ranked_candidates(
    query_vec: np.ndarray,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    qclient: Any,
    top_n: int,
    target_collection: str,
) -> tuple[list[int], dict[int, float]]:
    return dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=top_n,
        target_collection=target_collection,
    )"""

content = content.replace(old_dense_ranked, new_dense_ranked)

# 6. search function rewrite
old_search_sig = """def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
) -> list[dict]:"""

new_search_sig = """def _search_single_collection(
    query: str,
    top_k: int,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    collection: str,
) -> list[dict]:"""

content = content.replace(old_search_sig, new_search_sig)

# We need to replace `top_k = max(1, int(top_k))` which is in the old search
content = content.replace("top_k = max(1, int(top_k))", "")

# Inside _search_single_collection, replace `_dense_ranked_candidates` call
old_dense_call = """dense_ranked, dense_scores = _dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=dense_n,
    )"""

new_dense_call = """dense_ranked, dense_scores = _dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=dense_n,
        target_collection=collection,
    )"""

content = content.replace(old_dense_call, new_dense_call)

# Finally, inject the real `search` function right before `_format_source_label`
wrapper_search = """def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtro_area: str | Iterable[str] | None = None,
    filtro_assunto: str | Iterable[str] | None = None,
    target_collection: str | list[str] = COLLECTION_NAME,
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
        
    top_k = max(1, int(top_k))

    collections = target_collection if isinstance(target_collection, list) else [target_collection]
    
    all_results = []
    for col in collections:
        all_results.extend(_search_single_collection(
            query=query, 
            top_k=top_k, 
            filtro_area=filtro_area, 
            filtro_assunto=filtro_assunto, 
            collection=col
        ))
        
    if len(collections) > 1:
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:top_k]
        
    return all_results

def _format_source_label"""

content = content.replace("def _format_source_label", wrapper_search)

# Write back
with open(RETRIEVER_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied to retriever.py")
