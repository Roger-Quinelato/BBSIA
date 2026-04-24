# 🧠 Prompt de Implementação — API REST do Chatbot RAG BBSIA

> **Contexto:** Este prompt serve como especificação completa para um agente de IA ou desenvolvedor implementar a camada de API REST sobre o pipeline RAG BBSIA já existente.

---

## Contexto do Projeto

Você está trabalhando no projeto **BBSIA (Banco Brasileiro de Soluções de IA)**, um chatbot RAG 100% local (air-gapped) desenvolvido pelo LIIA (Laboratório de Inovação em Inteligência Artificial).

### Pipeline já implementado (não altere esses arquivos):

| Arquivo | Responsabilidade |
|---|---|
| `extrator_pdf_v2.py` | Extrai texto dos PDFs com PyMuPDF. Expõe `run_extraction()` |
| `chunking.py` | Divide o texto em chunks de 500 palavras com 75 de overlap. Expõe `run_chunking()` |
| `embedding.py` | Gera embeddings com `paraphrase-multilingual-mpnet-base-v2` e cria índice FAISS. Expõe `run_embedding()` |
| `rag_engine.py` | Motor RAG principal. Expõe as funções abaixo |
| `app.py` | Interface Streamlit (consumidora da API que você vai criar) |

### Funções do `rag_engine.py` que a API deve encapsular:

```python
# Busca semântica com filtros opcionais
search(query: str, top_k: int, filtro_area: list[str], filtro_assunto: list[str]) -> list[dict]

# Resposta completa (busca + LLM via Ollama)
answer_question(pergunta: str, model: str, top_k: int, filtro_area: list, filtro_assunto: list) -> dict
# Retorna: {"resposta": str, "fontes": list[str], "resultados": list[dict], "prompt": str}

# Listagens para popular dropdowns/filtros
list_available_areas() -> list[str]          # ex: ["arquitetura", "etica", "infraestrutura", ...]
list_available_assuntos() -> list[str]        # ex: ["RAG", "LGPD", "kubernetes", ...]
list_ollama_models() -> list[str]             # ex: ["llama3.1:8b", "mistral:7b"]

# Gerenciamento do índice em memória
reload_resources() -> None   # invalida o cache e recarrega o índice FAISS
```

### Estrutura de Chunks (cada item retornado por `search()`):

```json
{
  "score": 0.87,
  "id": 42,
  "documento": "LIIA BBSIA - Fase 1 MVP Banco Brasileiro de Soluções de IA v.02.pdf",
  "pagina": 12,
  "area": "desenvolvimento",
  "assuntos": ["MVP", "backend", "API", "cronograma"],
  "texto": "Trecho do documento...",
  "chunk_index": 3
}
```

### Áreas disponíveis na base:
`arquitetura`, `desenvolvimento`, `etica`, `governanca`, `infraestrutura`, `metodologia`

### Serviços externos que a API consome:
- **Ollama:** `http://localhost:11434` — LLM local (llama3.1:8b ou mistral:7b)
- **FAISS:** índice local em `faiss_index/index.faiss` + `faiss_index/metadata.json`

---

## Tarefa

Crie o arquivo `api.py` — uma **API REST com FastAPI** que exponha todos os recursos do pipeline RAG BBSIA para consumo externo (front-ends, integrações, testes automatizados).

---

## Especificação dos Endpoints

### 1. `POST /chat` — Pergunta ao chatbot (resposta completa)

**Request body:**
```json
{
  "pergunta": "Quais são os requisitos de infraestrutura do BBSIA?",
  "modelo": "llama3.1:8b",
  "top_k": 5,
  "filtro_area": ["infraestrutura"],
  "filtro_assunto": []
}
```

**Response `200`:**
```json
{
  "resposta": "O BBSIA requer...",
  "fontes": [
    "LIIA BBSIA - Infra-estrutura.pdf (p. 3)",
    "LIIA BBSIA - Infra-estrutura.pdf (p. 7)"
  ],
  "resultados": [
    {
      "score": 0.91,
      "id": 88,
      "documento": "LIIA BBSIA - Infra-estrutura.pdf",
      "pagina": 3,
      "area": "infraestrutura",
      "assuntos": ["servidores", "nuvem", "kubernetes"],
      "texto": "...",
      "chunk_index": 2
    }
  ],
  "modelo_usado": "llama3.1:8b",
  "total_fontes": 2,
  "total_chunks_recuperados": 5
}
```

**Campos obrigatórios no body:** apenas `pergunta`.  
**Defaults:** `modelo="llama3.1:8b"`, `top_k=5`, `filtro_area=[]`, `filtro_assunto=[]`.

---

### 2. `POST /search` — Busca semântica pura (sem LLM)

**Request body:**
```json
{
  "query": "infraestrutura kubernetes",
  "top_k": 5,
  "filtro_area": [],
  "filtro_assunto": []
}
```

**Response `200`:**
```json
{
  "query": "infraestrutura kubernetes",
  "total": 5,
  "resultados": [ /* lista de chunks igual ao /chat */ ]
}
```

---

### 3. `GET /areas` — Lista áreas disponíveis para filtro

**Response `200`:**
```json
{
  "areas": ["arquitetura", "desenvolvimento", "etica", "governanca", "infraestrutura", "metodologia"]
}
```

---

### 4. `GET /assuntos` — Lista assuntos disponíveis para filtro

**Response `200`:**
```json
{
  "assuntos": ["API", "LGPD", "MVP", "RAG", "avaliacao", "chatbot", "kubernetes", "regulacao"]
}
```

---

### 5. `GET /modelos` — Lista modelos Ollama disponíveis

**Response `200`:**
```json
{
  "modelos": ["llama3.1:8b", "mistral:7b"],
  "default": "llama3.1:8b"
}
```

---

### 6. `POST /reprocessar` — Reprocessa a base de conhecimento completa

Executa a cadeia: `run_extraction()` → `run_chunking()` → `run_embedding()` → `reload_resources()`.

**Response `200`:**
```json
{
  "status": "ok",
  "mensagem": "Base reprocessada com sucesso.",
  "etapas_concluidas": ["extracao", "chunking", "embedding", "reload"]
}
```

**Response `500` (se qualquer etapa falhar):**
```json
{
  "detail": "Falha na etapa 'chunking': [mensagem de erro]"
}
```

---

### 7. `POST /recarregar` — Recarrega o índice FAISS em memória (sem reprocessar)

**Response `200`:**
```json
{
  "status": "ok",
  "mensagem": "Indice FAISS recarregado em memoria."
}
```

---

### 8. `POST /upload` — Faz upload de um ou mais PDFs para `uploads/`

Aceita `multipart/form-data` com um ou mais arquivos `.pdf`.

**Response `200`:**
```json
{
  "arquivos_salvos": ["novo_documento.pdf"],
  "total": 1,
  "mensagem": "Arquivos salvos em uploads/. Execute /reprocessar para incluir na base."
}
```

**Rejeitar** arquivos que não sejam `.pdf` com HTTP 422.

---

### 9. `GET /status` — Health check e informações do sistema

**Response `200`:**
```json
{
  "status": "ok",
  "indice_carregado": true,
  "total_chunks": 347,
  "areas_disponiveis": 6,
  "assuntos_disponiveis": 24,
  "ollama_online": true,
  "modelos_disponiveis": ["llama3.1:8b"],
  "versao_api": "1.0.0"
}
```

---

## Requisitos Técnicos

### Framework e dependências

```python
# Usar FastAPI + Uvicorn
# pip install fastapi uvicorn python-multipart
```

### Estrutura do arquivo `api.py`

```
api.py
├── Imports e configurações
├── Inicialização do app FastAPI (com título, versão, descrição)
├── CORS configurado para aceitar qualquer origem (desenvolvimento local)
├── Pydantic models para todos os request/response bodies
├── Routers organizados por recurso (opcional, mas preferível)
└── Endpoints na ordem listada acima
```

### Inicialização

```python
app = FastAPI(
    title="BBSIA RAG API",
    description="API REST para o Chatbot RAG do Banco Brasileiro de Soluções de IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

### CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Pydantic Models obrigatórios

```python
class ChatRequest(BaseModel):
    pergunta: str
    modelo: str = "llama3.1:8b"
    top_k: int = Field(default=5, ge=1, le=20)
    filtro_area: list[str] = []
    filtro_assunto: list[str] = []

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filtro_area: list[str] = []
    filtro_assunto: list[str] = []

class ChunkResult(BaseModel):
    score: float
    id: int
    documento: str
    pagina: int | None
    area: str
    assuntos: list[str]
    texto: str
    chunk_index: int | None

class ChatResponse(BaseModel):
    resposta: str
    fontes: list[str]
    resultados: list[ChunkResult]
    modelo_usado: str
    total_fontes: int
    total_chunks_recuperados: int

class SearchResponse(BaseModel):
    query: str
    total: int
    resultados: list[ChunkResult]
```

### Tratamento de erros

- Capture `FileNotFoundError` do `rag_engine` (índice não encontrado) e retorne **HTTP 503** com mensagem clara: `"Índice FAISS não encontrado. Execute /reprocessar primeiro."`
- Capture `ConnectionError` / timeout do Ollama e retorne **HTTP 502** com mensagem: `"Ollama não está acessível em localhost:11434"`
- Para erros genéricos inesperados, retorne **HTTP 500** com o detalhe do erro

### Carregamento do índice

Ao iniciar o servidor, **pré-carregue** o índice FAISS usando `startup_event`:

```python
@app.on_event("startup")
async def startup_event():
    try:
        from rag_engine import _load_resources
        _load_resources()
        print("Índice FAISS carregado com sucesso.")
    except Exception as e:
        print(f"AVISO: Índice não carregado na inicialização: {e}")
```

### Execução

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
```

**Comando para rodar:**
```bash
python api.py
# ou
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Restrições e Regras

1. **NÃO modifique** nenhum dos arquivos existentes (`rag_engine.py`, `chunking.py`, `embedding.py`, `extrator_pdf_v2.py`, `app.py`)
2. **NÃO use async** nos endpoints que chamam `rag_engine` — as funções do motor são síncronas e bloqueantes. Use `run_in_executor` ou endpoints síncronos normais.
3. **Todos os textos** de resposta e mensagem devem estar em **português brasileiro**
4. **O endpoint `/chat`** deve ser tolerante a falhas do Ollama — se o LLM não responder, retorne os chunks recuperados com uma mensagem informando o problema, sem falhar completamente
5. **Upload de arquivos** deve validar extensão `.pdf` e rejeitar qualquer outro tipo com HTTP 422
6. **O campo `pergunta`** em `/chat` e `query` em `/search` são obrigatórios — retorne HTTP 422 se ausentes ou vazios (string com apenas espaços)

---

## Arquivos a criar

| Arquivo | Conteúdo |
|---|---|
| `api.py` | API FastAPI completa conforme especificação acima |
| `requirements.txt` | Adicionar `fastapi>=0.111.0`, `uvicorn>=0.30.0`, `python-multipart>=0.0.9` às dependências já existentes |

---

## Exemplo de uso esperado (curl)

```bash
# Health check
curl http://localhost:8000/status

# Pergunta ao chatbot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "O que é o BBSIA?", "top_k": 3}'

# Busca semântica sem LLM
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "kubernetes infraestrutura", "filtro_area": ["infraestrutura"]}'

# Upload de PDF
curl -X POST http://localhost:8000/upload \
  -F "files=@/caminho/para/novo_doc.pdf"

# Reprocessar base completa
curl -X POST http://localhost:8000/reprocessar
```

---

## Documentação automática

Após implementar, o Swagger UI estará disponível em:
- **`http://localhost:8000/docs`** — Swagger UI interativo
- **`http://localhost:8000/redoc`** — ReDoc

---

> [!IMPORTANT]
> O `app.py` (Streamlit) já existe e consome diretamente as funções do `rag_engine.py`. Após criar a API, o `app.py` pode ser refatorado (opcionalmente) para consumir a API via `requests`, mas isso **não é escopo desta tarefa**.
