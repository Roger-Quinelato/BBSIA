import os
import re

with open('retriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace _load_from_disk
load_from_disk_pattern = re.compile(r'    def _load_from_disk\(self\) -> dict:.*?        return \{', re.DOTALL)
load_from_disk_replacement = """    def _load_from_disk(self) -> dict:
        import json
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        
        base_dir = _script_dir()
        metadata_path = os.path.join(base_dir, INDEX_DIR, METADATA_FILE)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata nao encontrada em: {metadata_path}")

        qdrant_path = os.path.join(DATA_DIR, "qdrant_db")
        qclient = QdrantClient(path=qdrant_path)

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
            
        token_counts, doc_lengths, doc_freq, avgdl = _build_sparse_index(chunks)

        return {"""
content = load_from_disk_pattern.sub(load_from_disk_replacement, content)

# 2. Update the returned dict in _load_from_disk
dict_pattern = r'''            "index": index,
            "chunks": chunks,
            "model": model,
            "embeddings": embeddings,'''
dict_replacement = '''            "qclient": qclient,
            "chunks": chunks,
            "model": model,
            "embeddings": None,'''
content = content.replace(dict_pattern, dict_replacement)

# 3. Replace _dense_ranked_candidates completely
dense_ranked_pattern = re.compile(r'def _dense_ranked_candidates\(.*?\) -> tuple\[list\[int\], dict\[int, float\]\]:.*?    return ranked, score_map', re.DOTALL)
dense_ranked_replacement = """def _dense_ranked_candidates(
    query_vec: np.ndarray,
    filtro_area: str | Iterable[str] | None,
    filtro_assunto: str | Iterable[str] | None,
    qclient: Any,
    top_n: int,
) -> tuple[list[int], dict[int, float]]:
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    areas = list({_norm(v) for v in _as_list(filtro_area) if v.strip()})
    assuntos = list({_norm(v) for v in _as_list(filtro_assunto) if v.strip()})

    must_conditions = []
    if areas:
        must_conditions.append(
            FieldCondition(key="area", match=MatchAny(any=areas))
        )
    if assuntos:
        must_conditions.append(
            FieldCondition(key="assuntos", match=MatchAny(any=assuntos))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    try:
        results = qclient.search(
            collection_name="bbsia_chunks",
            query_vector=query_vec.tolist(),
            query_filter=query_filter,
            limit=top_n
        )
    except Exception as e:
        print("Qdrant search error:", e)
        return [], {}

    ranked = []
    score_map = {}
    for hit in results:
        doc_id = hit.id
        ranked.append(doc_id)
        score_map[doc_id] = float(hit.score)

    return ranked, score_map"""
content = dense_ranked_pattern.sub(dense_ranked_replacement, content)

# 4. Update cache_health
cache_health_pattern = re.compile(r'    index = data\.get\("index"\).*?    embedding_count = int\(getattr\(index, "ntotal", 0\) or 0\) if index is not None else 0', re.DOTALL)
cache_health_replacement = """    embedding_count = len(chunks) if chunks else 0
    embedding_dim = EXPECTED_EMBEDDING_DIM"""
content = cache_health_pattern.sub(cache_health_replacement, content)

# 5. Update search function variables unpacking
search_vars_pattern = r'''    index = data\["index"\]
    all_embeddings = data\["embeddings"\]'''
search_vars_replacement = r'''    qclient = data["qclient"]'''
content = content.replace(search_vars_pattern, search_vars_replacement)

# 6. Update search function call to _dense_ranked_candidates
search_call_pattern = r'''    dense_ranked, dense_scores = _dense_ranked_candidates\(
        query_vec=query_vec,
        eligible_ids=eligible_ids,
        chunks_len=len\(chunks\),
        index=index,
        all_embeddings=all_embeddings,
        top_n=dense_n,
    \)'''
search_call_replacement = r'''    dense_ranked, dense_scores = _dense_ranked_candidates(
        query_vec=query_vec,
        filtro_area=filtro_area,
        filtro_assunto=filtro_assunto,
        qclient=qclient,
        top_n=dense_n,
    )'''
content = re.sub(search_call_pattern, search_call_replacement, content)

# Also remove FAISS specific size validation
faiss_val_pattern = r'''    if int\(query_vec\.shape\[0\]\) != int\(index\.d\):.*?        \)'''
faiss_val_replacement = r'''    if int(query_vec.shape[0]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Dimensao da consulta incompativel. "
            f"Consulta={int(query_vec.shape[0])}; esperado={EXPECTED_EMBEDDING_DIM}. "
        )'''
content = re.sub(faiss_val_pattern, faiss_val_replacement, content, flags=re.DOTALL)

with open('retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Migration of retriever.py completed")
