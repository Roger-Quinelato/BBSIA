# BBSIA — Chatbot RAG Local

Chatbot RAG (Retrieval-Augmented Generation) 100% local para o projeto
**Banco Brasileiro de Soluções de IA (BBSIA)**.

## Arquitetura

```
docs/*.pdf
    │
    ▼
extrator_pdf_v2.py ──► classificador_artigo.py ──► data/biblioteca.json
    │                        (metadados: título, autores, ano, área)
    ▼
documentos_extraidos_v2.json
    │
    ▼
chunking.py       ──► chunks enriquecidos com metadados bibliográficos
    │
    ▼
embedding.py      ──► Qdrant local (`data/qdrant_db/`) + metadados (`data/faiss_index/`)
    │
    ▼
rag_engine.py
    ├── Dense retrieval   (Qdrant local / cosine)
    ├── Sparse retrieval  (BM25 nativo)
    ├── Fusão RRF         (Reciprocal Rank Fusion)
    ├── Re-ranking        (CrossEncoder, opcional)
    ├── Faithfulness check (anti-alucinação)
    └── Citações acadêmicas (Sobrenome, Ano — Título)
    │
    ▼
api.py            ──► REST API FastAPI + interface web em /web
```

### Módulos principais

| Módulo | Descrição |
|--------|-----------|
| `extrator_pdf_v2.py` | Extração de PDFs com PyMuPDF (texto, headings, tabelas, OCR fallback) |
| `classificador_artigo.py` | Classificação inteligente de documentos: heurísticas + LLM (Ollama) → `data/biblioteca.json` |
| `chunking.py` | Chunking parent-child com metadados bibliográficos (título, autores, ano) |
| `embedding.py` | Geração de embeddings + indexação dense no Qdrant local |
| `rag_engine.py` | Motor RAG híbrido (dense + sparse + RRF) com faithfulness check e citações ricas |
| `api.py` | API REST FastAPI com auth, rate limiting, upload e interface web |
| `config.py` | Configuração centralizada via `.env` (sem dependências externas) |

## Início rápido

### 1) Criar ambiente virtual

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

### 3) Baixar modelo Ollama

```bash
ollama pull qwen3.5:7b-instruct
```

### 4) Processar a base de documentos

```bash
# Extração + classificação + chunking + embedding em sequência:
python extrator_pdf_v2.py
python chunking.py
python embedding.py
```

Ou via API (após iniciar o servidor):

```bash
curl -X POST http://localhost:8000/reprocessar
```

### 5) Rodar a API FastAPI (inclui chatbot web)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Interface web: `http://localhost:8000/web`
- Documentação OpenAPI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuração por ambiente

Copie `.env.example` para `.env` e ajuste conforme necessário:

```bash
# LLM e embeddings
OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL=qwen3.5:7b-instruct
ALLOWED_LLM_MODELS=qwen3.5:7b-instruct
ALLOW_REMOTE_OLLAMA=false
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024
HF_LOCAL_FILES_ONLY=true
PRELOAD_RAG_ON_STARTUP=true
PRELOAD_RERANKER_ON_STARTUP=false
RAG_HEALTH_LOAD_ON_STATUS=false

# Parâmetros de retrieval
TOP_K=5
MAX_CONTEXT_CHUNKS=3
MAX_CHARS_PER_CHUNK=700
OLLAMA_TIMEOUT_SEC=300
OLLAMA_NUM_PREDICT=120
OLLAMA_NUM_CTX=8192

# Retrieval híbrido (dense + sparse + RRF)
HYBRID_DENSE_CANDIDATES=40
HYBRID_SPARSE_CANDIDATES=80
RRF_K=60
MIN_DENSE_SCORE_PERCENT=18

# Re-ranking local apos RRF (mais qualidade, mais latência)
ENABLE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATES=20
RERANKER_TOP_N=3
RERANKER_MAX_LENGTH=512

# CORS (origens separadas por vírgula)
CORS_ORIGINS=http://localhost,http://127.0.0.1,http://localhost:8000

# Autenticação (se definido, exige header X-API-Key)
API_KEY=
READ_API_KEY=
ADMIN_API_KEY=

# Rate limit por IP
RATE_LIMIT_REQUESTS=120
RATE_LIMIT_WINDOW_SEC=60

# Tamanho máximo de upload por arquivo (em MB)
MAX_UPLOAD_SIZE_MB=50
MAX_PDF_PAGES=300
MAX_PDF_EXTRACTED_CHARS=2000000
PDF_VALIDATION_TIMEOUT_SEC=30
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/status` | Status do sistema (índice, Ollama, reprocessamento) |
| `GET` | `/rag/health` | Health do cache RAG, modelos carregados e threshold atual (`?load=true` força carga) |
| `POST` | `/chat` | Pergunta com resposta RAG completa e citações acadêmicas |
| `POST` | `/search` | Busca semântica sem geração LLM |
| `GET` | `/biblioteca` | Catálogo de documentos com filtros (`?area=ia&ano_min=2020`) |
| `GET` | `/biblioteca/{id}` | Metadados completos de um documento específico |
| `GET` | `/filtros` | Valores únicos disponíveis para filtros dinâmicos |
| `GET` | `/areas` | Lista áreas disponíveis na base vetorial |
| `GET` | `/assuntos` | Lista assuntos disponíveis na base vetorial |
| `GET` | `/modelos` | Lista modelos Ollama disponíveis |
| `POST` | `/upload` | Upload de PDFs (com validação de tamanho) |
| `POST` | `/upload-metadata` | Cadastro de metadados para uploads |
| `POST` | `/reprocessar` | Reprocessa a base em background (não-bloqueante) |
| `POST` | `/recarregar` | Recarrega o índice vetorial (Qdrant) em memória |

## Upload com metadados (área/assunto)

```bash
# Upload com metadados no mesmo request:
curl -X POST http://localhost:8000/upload \
  -F "files=@/caminho/doc.pdf" \
  -F "area=infraestrutura" \
  -F "assuntos=kubernetes,seguranca"

# Ajuste posterior de metadados:
curl -X POST http://localhost:8000/upload-metadata \
  -H "Content-Type: application/json" \
  -d '{"documento":"uploads/doc.pdf","area":"infraestrutura","assuntos":["kubernetes","seguranca"]}'
```

Após o upload, execute `POST /reprocessar`. O progresso pode ser acompanhado via `GET /status`.

## Testes automatizados

```bash
# Sempre usar o venv (requer faiss-cpu instalado)
.venv\Scripts\python.exe -m pytest -v
```

Cobertura atual — **55 testes**:

| Arquivo | Testes | Escopo |
|---------|--------|--------|
| `test_classificador.py` | 14 | Título, autores, ano, tipo, fallback LLM |
| `test_catalogo_solucoes.py` | 5 | Schema JSON, validação do catálogo e materialização para embeddings |
| `test_api.py` | 16 | Status, health RAG, search, chat, reprocessar, upload/quarentena, `/biblioteca`, `/filtros`, autoria nos chunks |
| `test_extraction_chunking.py` | 7 | Extração PDF, chunking, metadados, fallback, uploads aprovados |
| `test_embedding.py` | 5 | `_load_chunks`: arquivo ausente, formato inválido, sucesso |
| `test_filters.py` | 1 | `_filter_ids` por área e assunto |
| `test_rag_engine.py` | 7 | Fallback Ollama, faithfulness check, deduplicação, cache health, preload, calibração e qualidade de retrieval |

## Scripts utilitários

| Script | Descrição |
|--------|-----------|
| `scripts/calibrar_threshold.py` | Calibra `MIN_DENSE_SCORE_PERCENT` com queries reais e salva estatísticas |
| `scripts/gerar_embeddings_solucoes.py` | Valida `catalogo/solucoes_piloto.json` e gera embeddings batch em `data/solucoes_faiss_index/` |
| `scripts/dev.ps1` | Atalhos de desenvolvimento no Windows (test, lint, format, typecheck, run, reprocess, solucoes-embedding) |

```bash
python scripts/calibrar_threshold.py
```

### Atalhos de desenvolvimento

PowerShell:

```bash
.\scripts\dev.ps1 -Task test
.\scripts\dev.ps1 -Task lint
.\scripts\dev.ps1 -Task run
.\scripts\dev.ps1 -Task reprocess
.\scripts\dev.ps1 -Task solucoes-embedding
```

Makefile:

```bash
make test
make lint
make run
make reprocess
make solucoes-embedding
```

## Catálogo de soluções piloto

O projeto inclui um catálogo estruturado para soluções piloto:

- Schema JSON: `schemas/solucao_piloto.schema.json`.
- Dados piloto: `catalogo/solucoes_piloto.json`.
- Validador/materializador: `catalogo_solucoes.py`.
- Embeddings do catálogo: `scripts/gerar_embeddings_solucoes.py`.

O schema mínimo possui 10 campos obrigatórios:

| Campo | Finalidade |
|-------|------------|
| `id` | Identificador estável da solução |
| `nome` | Nome legível |
| `descricao` | Resumo da solução |
| `orgao` | Órgão/equipe responsável |
| `area` | Área temática controlada |
| `problema` | Problema público/operacional |
| `solucao` | Descrição da abordagem |
| `tecnologias` | Stack principal |
| `status` | Etapa da solução |
| `conformidade` | Bloco LGPD, soberania, modelo e open-source |

Para validar e gerar embeddings batch do catálogo:

```bash
python catalogo_solucoes.py
python scripts/gerar_embeddings_solucoes.py
```

O primeiro comando gera `data/solucoes_piloto_chunks.json`. O segundo também cria `data/solucoes_faiss_index/`.

## Conformidade

A revisão inicial está documentada em `CONFORMIDADE.md`.

Resumo:

- LGPD: o schema exige declaração de dados pessoais e base legal por solução.
- Soberania: o piloto privilegia execução local (`ALLOW_REMOTE_OLLAMA=false`, `HF_LOCAL_FILES_ONLY=true`).
- Open-source: o schema exige licença do modelo e lista de dependências principais.
- Produção: soluções com `dados_pessoais` ou `hospedagem` como `a_confirmar` devem passar por revisão antes de uso real.

## Versionamento de dados

- `data/` e `uploads/` ficam fora do Git (cache/artefatos de execução).
- `docs/` também fica fora do Git neste repositório. Ela contém PDFs/planilhas de referência usados para gerar a base RAG.
- `data/qdrant_db/`, `data/faiss_index/`, `data/chunks.json`, `data/biblioteca.json` e `data/documentos_extraidos_v2.json` são artefatos gerados localmente pelo pipeline.
- Se o projeto for compartilhado por `git clone`, o colega precisa receber os arquivos de `docs/` por outro canal ou adicionar seus próprios PDFs antes de rodar o pipeline.
- Se o projeto for compartilhado por `.zip` da pasta inteira, incluir `docs/` e, opcionalmente, `data/` para evitar reprocessamento inicial.

## Testar em outro notebook

Há dois caminhos suportados:

### Caminho A — clone limpo + reprocessamento

Use quando o colega vai clonar o repositório e gerar a base no próprio notebook.

Pré-requisitos:

- Python 3.10+.
- Ollama instalado e rodando.
- Modelo LLM local baixado:

```bash
ollama pull qwen3.5:7b-instruct
```

Passos:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Antes de reprocessar, coloque os PDFs/planilhas de referência em `docs/`.

Se os modelos Hugging Face ainda não estiverem no cache local, ajuste temporariamente o `.env`:

```bash
HF_LOCAL_FILES_ONLY=false
```

Depois gere a base:

```bash
python extrator_pdf_v2.py
python chunking.py
python embedding.py
uvicorn api:app --host 0.0.0.0 --port 8000
```

Validação:

```bash
.venv\Scripts\python.exe -m pytest -q
curl http://localhost:8000/status
curl http://localhost:8000/rag/health
```

### Caminho B — pacote completo da pasta

Use quando a prioridade é o colega testar rápido.

Ao criar o `.zip`, inclua:

- código do projeto;
- `docs/`;
- `data/`, se quiser que ele já tenha índice/chunks prontos;
- `.env.example`.

Não inclua:

- `.venv/`;
- `.env` com chaves locais;
- `uploads/quarantine/` com arquivos sensíveis ou pendentes de revisão.

Mesmo com `data/` incluído, o colega ainda precisa instalar dependências e ter os modelos locais compatíveis. Se mudar `EMBEDDING_MODEL` ou `EMBEDDING_DIM`, é obrigatório rodar `/reprocessar` ou executar novamente `extrator_pdf_v2.py`, `chunking.py` e `embedding.py`.
