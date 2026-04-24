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
embedding.py      ──► data/faiss_index/
    │
    ▼
rag_engine.py
    ├── Dense retrieval   (FAISS / Inner Product)
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
| `embedding.py` | Geração de embeddings + índice FAISS persistente |
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
ollama pull llama3.1:8b
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
DEFAULT_MODEL=llama3.1:8b
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Parâmetros de retrieval
TOP_K=5
MAX_CONTEXT_CHUNKS=3
MAX_CHARS_PER_CHUNK=700
OLLAMA_TIMEOUT_SEC=300
OLLAMA_NUM_PREDICT=120
OLLAMA_NUM_CTX=2048

# Retrieval híbrido (dense + sparse + RRF)
HYBRID_DENSE_CANDIDATES=40
HYBRID_SPARSE_CANDIDATES=80
RRF_K=60

# Re-ranking opcional (mais qualidade, mais latência)
ENABLE_RERANKER=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_CANDIDATES=20

# CORS (origens separadas por vírgula)
CORS_ORIGINS=http://localhost,http://127.0.0.1,http://localhost:8000

# Autenticação (se definido, exige header X-API-Key)
API_KEY=

# Rate limit por IP
RATE_LIMIT_REQUESTS=120
RATE_LIMIT_WINDOW_SEC=60

# Tamanho máximo de upload por arquivo (em MB)
MAX_UPLOAD_SIZE_MB=50
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/status` | Status do sistema (índice, Ollama, reprocessamento) |
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
| `POST` | `/recarregar` | Recarrega o índice FAISS em memória |

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

Cobertura atual — **40 testes**:

| Arquivo | Testes | Escopo |
|---------|--------|--------|
| `test_classificador.py` | 14 | Título, autores, ano, tipo, fallback LLM |
| `test_api.py` | 11 | Status, search, chat, reprocessar, upload, `/biblioteca`, `/filtros`, autoria nos chunks |
| `test_extraction_chunking.py` | 6 | Extração PDF, chunking, metadados, fallback |
| `test_embedding.py` | 5 | `_load_chunks`: arquivo ausente, formato inválido, sucesso |
| `test_filters.py` | 1 | `_filter_ids` por área e assunto |
| `test_rag_engine.py` | 3 | Fallback Ollama, faithfulness check, deduplicação |

## Scripts utilitários

| Script | Descrição |
|--------|-----------|
| `scripts/calibrar_threshold.py` | Calibra `MIN_DENSE_SCORE_PERCENT` com queries reais e salva estatísticas |

```bash
python scripts/calibrar_threshold.py
```
