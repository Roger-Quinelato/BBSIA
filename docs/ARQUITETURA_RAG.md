# Arquitetura RAG Consolidada (Qdrant Core)

Este documento descreve a arquitetura real consolidada das Fases 1 e 2 do BBSIA.

## 1) Resumo executivo

- Backend vetorial oficial: Qdrant local.
- Retrieval hibrido: dense (Qdrant) + sparse (BM25 local).
- Fusao de ranking: RRF.
- Re-ranking: opcional com cross-encoder.
- Geracao: Ollama local.
- Faithfulness no pipeline sincronico; no streaming, evento opcional via
  `ENABLE_STREAM_FAITHFULNESS`.
- Calibracao inicial registrada, com `MIN_DENSE_SCORE_PERCENT=18` mantido por
  decisao documentada.

## 2) Pipeline oficial de ingestao

```text
extracao PDF
  -> classificacao/metadados
  -> chunking estruturado
  -> embeddings
  -> persistencia no Qdrant local
```

Componentes principais:

1. `extrator_pdf_v2.py`
- Extrai texto/elementos em formato estruturado.
- Salva `data/documentos_extraidos_v2.json`.
- Aciona classificacao por payload e atualiza biblioteca.

2. `classificador_artigo.py`
- Classifica e enriquece metadados (heuristica + LLM opcional).
- Atualiza `data/biblioteca.json`.

3. `chunking.py`
- Consome `data/documentos_extraidos_v2.json`.
- Gera `data/chunks.json` com parent-child e metadados.

4. `embedding.py`
- Gera embeddings e indexa no Qdrant local.
- Atualiza metadados de indice em `data/qdrant_index_metadata/`.

## 3) Artefatos gerados

Core runtime:

- `data/documentos_extraidos_v2.json`
- `data/biblioteca.json`
- `data/chunks.json`
- `data/qdrant_db/`
- `data/qdrant_index_metadata/metadata.json`
- `data/qdrant_index_metadata/manifest.json`

Compatibilidade historica (temporaria):

- `data/faiss_index/metadata.json`
  - ainda pode ser lido como fallback de metadados, se presente;
  - nao e backend vetorial ativo.

## 4) Retrieval online

`retriever.py` / `rag_engine.py` aplicam o fluxo:

1. Dense retrieval no Qdrant local.
2. Sparse retrieval BM25 local (indice em memoria a partir de `chunks`).
3. Fusao por RRF (`RRF_K`).
4. Re-ranking opcional (`ENABLE_RERANKER`).
5. Montagem de contexto com citacoes e envio para geracao.

## 5) Geracao e faithfulness

`pipeline.py`:

- Caminho sincronico (`answer_question`):
  - gera resposta;
  - aplica controle de fidelidade no retorno final.

- Caminho streaming (`answer_question_stream`):
  - eventos base: `metadata`, `token`, `error`;
  - quando `ENABLE_STREAM_FAITHFULNESS=true`, emite evento final `faithfulness` com:
    - `faithfulness_checked`
    - `faithful`
    - `reason`
    - `fallback_response` (somente quando nao fiel)

## 6) Configuracoes relevantes

- Retrieval:
  - `TOP_K`
  - `HYBRID_DENSE_CANDIDATES`
  - `HYBRID_SPARSE_CANDIDATES`
  - `RRF_K`
  - `MIN_DENSE_SCORE_PERCENT`

Nota de calibracao: a evidencia mais recente esta registrada em
`docs/AVALIACAO_RAG.md` e em
`benchmarks/results/threshold_calibration_latest.json`. Na execucao de
2026-04-29, os scores dense ficaram zerados para todas as queries de
calibracao; por isso, a sugestao bruta de reduzir o threshold para `1` foi
tratada como nao acionavel, e o valor `MIN_DENSE_SCORE_PERCENT=18` foi mantido.

- Reranker:
  - `ENABLE_RERANKER`
  - `RERANKER_MODEL`
  - `RERANKER_CANDIDATES`
  - `RERANKER_TOP_N`

- Streaming faithfulness:
  - `ENABLE_STREAM_FAITHFULNESS` (default: `false`)

## 7) Dados locais, uploads e caches

- `data/` e `uploads/` sao artefatos locais de execucao.
- Caches de modelos (HF/Ollama) dependem do host e ambiente.
- Esses artefatos nao representam contrato de API publica, e sim estado operacional local.

## 8) Escopo opcional fora do core runtime

- `catalogo/`, `schemas/`, `benchmarks/` e scripts associados permanecem como trilha opcional.
- Podem conter referencias historicas de FAISS em dados de catalogo/benchmark, sem impacto no backend vetorial oficial do runtime.
- LangChain, RAGAS e DeepEval sao ferramentas opcionais de avaliacao, nao
  dependencias do runtime consolidado.
- Redis, Docker, Helm, OIDC/RBAC, auditoria em banco, Elasticsearch, grafo e
  fine-tuning ficam fora da Fase 2 consolidada; entram somente na Fase 3 ou
  posterior, mediante baseline salvo e decisao documentada.

## 9) Marco final da Fase 2 e entrada da Fase 3

A Fase 2 considera o runtime consolidado quando a suite completa e os smoke
imports passam, a calibracao inicial esta registrada e o comportamento padrao
do retrieval permanece inalterado. O benchmark expandido nao e uma pendencia
retroativa da Fase 2; ele e o primeiro gate da Fase 3.

Antes de qualquer mudanca estrutural da Fase 3, registrar um baseline conforme
`docs/PLANO_BENCHMARK_RAG_FASE3.md`. Esse baseline deve medir retrieval,
geracao, grounding, out-of-scope e latencia, mantendo a arquitetura atual como
comparador oficial.

Nenhuma troca estrutural de backend, retrieval, geracao, avaliador ou
infraestrutura deve ser feita antes de salvar esse baseline em
`benchmarks/results/`.
