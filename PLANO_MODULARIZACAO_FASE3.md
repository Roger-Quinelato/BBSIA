# Plano de Modularizacao da Fase 3

Este documento prepara a Fase 3 do BBSIA sem alterar a estrutura atual de
runtime. A consolidacao das Fases 1 e 2 deve permanecer estavel: API, pipeline
RAG, Qdrant local, busca hibrida, reranker e validacoes atuais continuam
funcionando nos mesmos pontos de entrada enquanto a modularizacao e planejada.

## Objetivo

A Fase 3 deve transformar a arquitetura atual em um pacote mais claro e
testavel, sem quebrar imports publicos nem mudar comportamento de producao em
um unico passo. A migracao deve ser incremental, com fachadas temporarias para
preservar compatibilidade.

## Estrutura alvo

```text
bbsia/
  app/
    api.py
    api_core.py
    routers/
      admin.py
      biblioteca.py
      rag.py
      system.py
  rag/
    ingestion.py
    chunking.py
    embeddings.py
    vector_store.py
    retrieval.py
    generation.py
    faithfulness.py
    query_planning.py
  config/
    settings.py
```

### Responsabilidades previstas

| Modulo futuro | Responsabilidade |
| --- | --- |
| `bbsia/app/` | API FastAPI, middlewares, upload, status e routers. |
| `bbsia/rag/ingestion.py` | Orquestracao de extracao, classificacao e metadados de documentos. |
| `bbsia/rag/chunking.py` | Chunking estruturado parent-child a partir de `documentos_extraidos_v2.json`. |
| `bbsia/rag/embeddings.py` | Geracao de embeddings e materializacao de metadados do indice. |
| `bbsia/rag/vector_store.py` | Contrato com Qdrant local e health do backend vetorial. |
| `bbsia/rag/retrieval.py` | Busca hibrida: dense via Qdrant, sparse local, RRF, filtros e deduplicacao. |
| `bbsia/rag/generation.py` | Prompting, validacao de modelo e chamadas ao Ollama. |
| `bbsia/rag/faithfulness.py` | Checagens de grounding e fallback extrativo. |
| `bbsia/rag/query_planning.py` | Base futura para self-query, inferencia de filtros e reescrita de consulta. |
| `bbsia/config/settings.py` | Configuracao agrupada por dominio, carregada apos `.env`. |

## Estado atual preservado

Por enquanto, os modulos atuais continuam sendo os pontos estaveis do sistema:

- `api.py` continua sendo o entrypoint FastAPI.
- `api_core.py` continua concentrando app, middlewares e helpers compartilhados.
- `rag_engine.py` continua como fachada historica do motor RAG.
- `retriever.py` continua expondo `search()` e utilitarios de recuperacao.
- `pipeline.py` continua expondo `answer_question()` e `answer_question_stream()`.
- `generator.py`, `reranker.py` e `faithfulness.py` continuam como modulos
  funcionais de geracao, reranking e checagem.
- `embedding.py`, `chunking.py` e `extrator_pdf_v2.py` continuam como etapas do
  pipeline de reprocessamento.

Essa decisao evita reorganizacao ampla antes da Fase 3. A migracao futura deve
primeiro criar novos modulos, depois apontar as fachadas antigas para eles, e so
por fim remover wrappers obsoletos.

## Imports publicos obrigatorios

Estes imports devem continuar funcionando durante toda a transicao:

```python
import api
import rag_engine
from rag_engine import _filter_ids
from retriever import search
from pipeline import answer_question
```

Qualquer mudanca que quebre esses imports deve ser tratada como regressao, a
menos que haja uma etapa explicita de remocao com testes e documentacao de
migracao.

## Sequencia segura de migracao

1. Criar pacote `bbsia/` sem mover arquivos existentes.
2. Adicionar modulos novos com codigo extraido em fatias pequenas.
3. Manter os arquivos atuais como fachadas temporarias.
4. Migrar testes para cobrir tanto fachada antiga quanto modulo novo.
5. Atualizar imports internos gradualmente.
6. Remover wrappers apenas depois de uma janela de compatibilidade.

## Guardrails da Fase 3

- Nao misturar reorganizacao de pacotes com mudanca de ranking ou geracao.
- Nao alterar o backend vetorial oficial: Qdrant local permanece o core.
- Nao reintroduzir FAISS como dependencia de runtime core.
- Nao mudar formatos gerados (`chunks.json`, `biblioteca.json`,
  `documentos_extraidos_v2.json`) sem migracao explicita.
- Nao sobrescrever filtros explicitos do usuario ao introduzir query planning.
- Manter smoke imports e testes de API em toda etapa.

## Validacao minima por etapa

```powershell
@'
import api
import rag_engine
from rag_engine import _filter_ids
from retriever import search
from pipeline import answer_question
print("smoke ok")
'@ | .\.venv\Scripts\python.exe -
```

Para mudancas que toquem recuperacao, pipeline ou API, tambem rodar:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_api.py tests/test_filters.py tests/test_rag_engine.py tests/test_pipeline_integration.py
```

