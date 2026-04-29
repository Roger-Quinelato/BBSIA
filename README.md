# BBSIA - Chatbot RAG Local

Backend RAG local para o projeto Banco Brasileiro de Solucoes de IA (BBSIA).

## Estado consolidado (Fases 1 e 2)

- Backend vetorial oficial: Qdrant local.
- Core runtime: sem FAISS para busca vetorial.
- Fase 2 encerrada como runtime estavel: Qdrant local, BM25 local, RRF,
  reranker opcional, Ollama local e faithfulness no pipeline.
- Calibracao inicial registrada em
  `benchmarks/results/threshold_calibration_latest.json`, com
  `MIN_DENSE_SCORE_PERCENT=18` mantido por decisao documentada.
- Pipeline oficial:
  1. extracao PDF
  2. classificacao/metadados
  3. chunking estruturado
  4. embeddings
  5. Qdrant local
  6. retrieval hibrido
  7. reranker opcional
  8. geracao Ollama
  9. faithfulness no pipeline

## Fluxo de ingestao

```text
extrator_pdf_v2.py
  -> classificador_artigo.py
  -> chunking.py
  -> embedding.py
```

Artefatos principais:

- `data/documentos_extraidos_v2.json`: payload estruturado da extracao.
- `data/biblioteca.json`: catalogo de metadados dos documentos.
- `data/chunks.json`: chunks parent-child com metadados enriquecidos.
- `data/qdrant_db/`: armazenamento local do Qdrant (vetores + payloads).
- `data/qdrant_index_metadata/metadata.json`: metadados do indice ativo.
- `data/qdrant_index_metadata/manifest.json`: manifesto/hash do metadata.

Compatibilidade temporaria (legado):

- O runtime ainda aceita fallback de leitura em `data/faiss_index/metadata.json`
  quando `data/qdrant_index_metadata/metadata.json` nao existe.
- Esse fallback e apenas de compatibilidade historica e nao representa backend atual.

## Retrieval e geracao

- Dense retrieval: Qdrant local (cosine).
- Sparse retrieval: BM25 local em memoria.
- Fusao: RRF (Reciprocal Rank Fusion).
- Re-ranking: cross-encoder opcional (`ENABLE_RERANKER`).
- Geracao: Ollama local.
- Faithfulness:
  - resposta sincronica aplica controle de fidelidade no pipeline;
  - streaming usa evento opcional quando `ENABLE_STREAM_FAITHFULNESS=true`.

## API e streaming

- `POST /chat` (NDJSON):
  - eventos base: `metadata`, `token`, `error`.
  - evento extra opcional: `faithfulness` (somente com `ENABLE_STREAM_FAITHFULNESS=true`).
- `POST /search`: busca sem geracao.
- `POST /reprocessar`: roda o pipeline completo.
- `POST /recarregar`: recarrega recursos RAG em memoria.

## Configuracao relevante (.env)

- `OLLAMA_URL`
- `DEFAULT_MODEL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `HF_LOCAL_FILES_ONLY`
- `TOP_K`
- `HYBRID_DENSE_CANDIDATES`
- `HYBRID_SPARSE_CANDIDATES`
- `RRF_K`
- `MIN_DENSE_SCORE_PERCENT`
- `ENABLE_RERANKER`
- `RERANKER_MODEL`
- `RERANKER_CANDIDATES`
- `RERANKER_TOP_N`
- `ENABLE_STREAM_FAITHFULNESS`

## Desenvolvimento rapido

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python extrator_pdf_v2.py
python chunking.py
python embedding.py

uvicorn api:app --host 0.0.0.0 --port 8000
```

## Testes

Use sempre o Python da venv:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Dados e caches locais

`data/`, `uploads/` e caches locais sao artefatos de execucao.

- `data/`: indices, metadados, chunks e saidas de pipeline.
- `uploads/`: staging/aprovacao de arquivos enviados.
- caches de modelos (Hugging Face/Ollama) dependem do ambiente local.

## Catalogo e benchmarks (escopo opcional)

- `catalogo/`, `schemas/` e `benchmarks/` permanecem no repositorio como componentes opcionais.
- `scripts/gerar_embeddings_solucoes.py` gera artefatos do catalogo em `data/solucoes_faiss_index/` (legado/opcional para catalogo, fora do core runtime).
- O benchmark expandido em `benchmarks/` e o primeiro gate da Fase 3, nao uma
  pendencia retroativa da Fase 2.
- Nenhuma troca estrutural do RAG deve ocorrer antes de salvar baseline em
  `benchmarks/results/`.
- LangChain, RAGAS e DeepEval podem ser usados como avaliacao opcional em
  ambiente separado, mas nao sao dependencia do runtime.
- Redis, Docker, Helm, OIDC/RBAC, auditoria em banco, Elasticsearch, grafo e
  fine-tuning pertencem a Fase 3 ou posterior.

## Documentacao tecnica

- Arquitetura RAG consolidada: `docs/ARQUITETURA_RAG.md`.
- Avaliacao e calibracao: `docs/AVALIACAO_RAG.md`.
- Gate inicial da Fase 3: `docs/PLANO_BENCHMARK_RAG_FASE3.md`.
- Documentos de plano/prompt antigos foram mantidos apenas como referencia historica e nao devem ser usados como fonte do runtime atual.
