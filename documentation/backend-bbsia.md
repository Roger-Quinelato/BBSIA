# Backend BBSIA

Este guia explica a pasta `bbsia/`, que contem o backend Python principal do
repositorio. O restante do projeto vive fora desse pacote: `docs/`, `data/`,
`uploads/`, `tests/`, `RPI/`, `Makefile`, `.env.example` e arquivos de
configuracao.

## Papel da pasta `bbsia/`

A pasta `bbsia/` implementa a API, o motor RAG, a ingestao de documentos, o
catalogo de solucoes piloto, os benchmarks e os comandos operacionais. Ela e o
pacote Python instalado pelo projeto `bbsia-rag`.

## Camadas principais

```text
bbsia/
|-- app/             # API FastAPI, contratos, runtime, uploads e seguranca
|-- cli/             # Comandos operacionais usados por humanos e Makefile
|-- core/            # Configuracao, categorias e observabilidade
|-- domain/          # Regras de dominio para catalogo, biblioteca e metadados
|-- evaluation/      # Benchmarks de qualidade do RAG
|-- infrastructure/  # Integracoes tecnicas compartilhadas
`-- rag/             # Ingestao, retrieval, geracao e orquestracao RAG
```

## `app/`

`bbsia/app/` expoe a API HTTP com FastAPI.

- `bootstrap/main.py` cria o ponto de entrada e registra os routers.
- `runtime/app.py` configura FastAPI, CORS, metricas, middleware de seguranca,
  ciclo de vida e arquivos estaticos em `/web`.
- `contracts/schemas.py` define os modelos Pydantic usados pelos endpoints.
- `routers/rag.py` expoe chat, busca, areas, assuntos, modelos e saude do RAG.
- `routers/admin.py` expoe upload, quarentena, aprovacao, recarga e
  reprocessamento.
- `routers/biblioteca.py` expoe a biblioteca de documentos.
- `routers/system.py` expoe `/`, `/status` e dados operacionais.
- `security/auth.py` aplica API key opcional e rate limit por IP.

Quando `API_KEY`, `READ_API_KEY` ou `ADMIN_API_KEY` ficam vazias, os endpoints
correspondentes aceitam chamadas sem chave. Quando voce configura chaves, envie
o header `X-API-Key`.

## `core/`

`bbsia/core/config.py` carrega `.env` sem depender de bibliotecas externas e
materializa as configuracoes em dataclasses. As principais areas de
configuracao sao:

- autenticacao, upload e rate limit;
- URL, modelo e limites do Ollama;
- modelo de embedding e dimensao esperada;
- parametros de retrieval hibrido;
- reranker opcional;
- preload do RAG na inicializacao.

`bbsia/core/categorias.yaml` funciona como fallback manual para categorias de
documentos.

## `domain/`

`bbsia/domain/` concentra regras de dominio que nao pertencem diretamente a
FastAPI nem ao motor RAG.

- `catalogo/` valida `solucoes_piloto.json` com JSON Schema e materializa
  chunks para indexacao.
- `document_library/` carrega a biblioteca de documentos e alimenta filtros da
  API.
- `document_metadata/` normaliza e persiste metadados de uploads.

O catalogo de solucoes piloto e a base do modo diagnostico. Quando a pergunta
descreve sintomas, falhas ou problemas, o pipeline busca solucoes candidatas e
usa documentos apenas como evidencias de apoio.

## `infrastructure/`

`bbsia/infrastructure/vector_store.py` isola a integracao com Qdrant local. O
codigo usa duas colecoes:

- `bbsia_chunks`: documentos indexados.
- `bbsia_solucoes`: solucoes piloto materializadas como chunks.

A busca densa aceita filtros de `area` e `assuntos` quando eles sao enviados
pela API ou inferidos pelo query planning.

## `rag/`

`bbsia/rag/` implementa o motor RAG.

### Ingestao

`rag/ingestion/` executa o pipeline de documentos:

1. Liste PDFs da raiz do repositorio, de `docs/` e de `uploads/approved`.
2. Extraia texto, secoes e tabelas com Docling quando disponivel.
3. Use PyMuPDF e OCR como fallback quando necessario.
4. Grave `data/documentos_extraidos_v2.json`.
5. Gere chunks parent-child em `data/chunks.json` e `data/parents.json`.
6. Gere embeddings com prefixo E5 `passage:`.
7. Atualize o Qdrant local e os metadados em `data/qdrant_index_metadata/`.

### Recuperacao

`rag/retrieval/` combina:

- busca densa no Qdrant;
- busca lexical BM25 local;
- Reciprocal Rank Fusion (RRF);
- deduplicacao por parent chunk;
- reranker opcional com CrossEncoder;
- query planning opcional para inferir area, assunto, ano ou tipo.

O parametro `MIN_DENSE_SCORE_PERCENT` evita respostas quando a recuperacao nao
tem sinal suficiente de evidencia.

### Geracao

`rag/generation/` monta prompts e consulta o Ollama. O prompt instrui o modelo a
responder em portugues, usar somente o contexto fornecido e citar fontes no
formato `(Sobrenome, Ano)` quando aplicavel.

O modo diagnostico exige secoes fixas:

- Diagnostico
- Solucao Recomendada
- Passos
- Riscos

Se nao houver solucao candidata no catalogo, o sistema retorna uma resposta
conservadora em vez de inventar passos de implantacao.

### Orquestracao

`rag/orchestration/pipeline.py` decide se a pergunta e documental ou
diagnostica, chama os dominios de recuperacao corretos, monta contexto, invoca o
Ollama e aplica fallback quando falta evidencia ou quando a geracao falha.

## `evaluation/`

`bbsia/evaluation/benchmarks/` contem scripts para medir qualidade do RAG.

- `generate_eval.py` gera perguntas de avaliacao a partir de chunks.
- `benchmark.py` executa avaliacao com RAGAS quando as dependencias estao
  disponiveis.
- `rag_benchmark.py` executa benchmark com metricas heuristicas e suporte
  opcional a DeepEval.

As metricas cobrem `context_recall`, `faithfulness_score`,
`answer_relevancy` e `solution_match`.

## `cli/`

`bbsia/cli/` contem utilitarios operacionais, como:

- calibrar threshold de recuperacao;
- conversar com o chatbot pelo terminal;
- diagnosticar Qdrant;
- gerar embeddings de solucoes;
- perguntar aos documentos.

Use esses comandos para operacao local e diagnostico durante desenvolvimento.
