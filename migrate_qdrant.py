import os

with open('embedding.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    LOGGER.info("event=embedding_index_building dim=%s", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)"""

replacement = """    LOGGER.info("event=embedding_index_building dim=%s", dim)
    
    qdrant_path = os.path.join(DATA_DIR, "qdrant_db")
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    
    client = QdrantClient(path=qdrant_path)
    collection_name = "bbsia_chunks"
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload=chunk
            )
        )
    
    batch_size_points = 500
    for j in range(0, len(points), batch_size_points):
        client.upload_points(collection_name=collection_name, points=points[j:j+batch_size_points])

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        f.write("QDRANT_MIGRATED")"""

if target in content:
    with open('embedding.py', 'w', encoding='utf-8') as f:
        f.write(content.replace(target, replacement))
    print('Replaced successfully')
else:
    target = target.replace('\\n', '\\r\\n')
    if target in content:
        with open('embedding.py', 'w', encoding='utf-8') as f:
            f.write(content.replace(target, replacement))
        print('Replaced successfully with CRLF')
    else:
        print('Target not found in embedding.py')
