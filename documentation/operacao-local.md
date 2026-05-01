# Operacao local

Este guia descreve como executar, reprocessar e diagnosticar o BBSIA RAG em uma
maquina local.

## Preparar ambiente

Crie o ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependencias:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Copie o arquivo de exemplo:

```powershell
Copy-Item .env.example .env
```

Edite `.env` antes de subir a API.

## Configurar Ollama

Configure a URL local:

```env
OLLAMA_URL=http://localhost:11434
```

Defina o modelo padrao e a lista de modelos permitidos:

```env
DEFAULT_MODEL=qwen3.5:7b-instruct
ALLOWED_LLM_MODELS=qwen3.5:7b-instruct
```

Confira os modelos instalados:

```powershell
ollama list
```

Por padrao, a aplicacao bloqueia Ollama remoto:

```env
ALLOW_REMOTE_OLLAMA=false
```

Mantenha esse valor em `false` para uso local. Altere apenas quando houver uma
decisao explicita de infraestrutura e seguranca.

## Configurar embeddings

O pipeline usa `intfloat/multilingual-e5-large` por padrao:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024
HF_LOCAL_FILES_ONLY=true
```

Quando `HF_LOCAL_FILES_ONLY=true`, o modelo precisa existir no cache local. Use
`HF_LOCAL_FILES_ONLY=false` apenas para preparar o ambiente ou baixar modelos em
uma maquina com acesso a rede.

## Configurar retrieval

Parametros principais:

```env
TOP_K=5
MAX_CONTEXT_CHUNKS=6
HYBRID_DENSE_CANDIDATES=40
HYBRID_SPARSE_CANDIDATES=80
RRF_K=60
MIN_DENSE_SCORE_PERCENT=18
```

O retrieval combina busca densa no Qdrant, busca lexical BM25 e RRF. O valor
`MIN_DENSE_SCORE_PERCENT` controla quando o sistema deve responder com fallback
por falta de evidencia.

## Configurar reranker

O reranker melhora qualidade, mas aumenta latencia e precisa de modelo local ou
acesso para download.

```env
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_CANDIDATES=20
RERANKER_TOP_N=3
RERANKER_MAX_LENGTH=512
```

Desative se o modelo nao estiver disponivel:

```env
ENABLE_RERANKER=false
```

## Subir a API

Execute:

```powershell
.\.venv\Scripts\uvicorn.exe bbsia.app.bootstrap.main:app --host 0.0.0.0 --port 8000
```

Ou use:

```powershell
make run
```

Verifique a saude:

```powershell
Invoke-RestMethod http://localhost:8000/status
```

## Reprocessar documentos

Use o endpoint administrativo:

```powershell
make reprocess
```

O reprocessamento executa estas etapas:

1. Liste PDFs na raiz do repositorio, em `docs/` e em `uploads/approved`.
2. Extraia texto, secoes, tabelas e OCR quando necessario.
3. Grave `data/documentos_extraidos_v2.json`.
4. Gere chunks em `data/chunks.json`.
5. Grave parents em `data/parents.json`.
6. Gere embeddings.
7. Atualize o Qdrant local e metadados em `data/qdrant_index_metadata/`.

Consulte o estado em:

```powershell
Invoke-RestMethod http://localhost:8000/status
```

## Processar solucoes piloto

O catalogo vive em:

```text
bbsia/domain/catalogo/data/solucoes_piloto.json
```

Valide e materialize chunks pelo dominio de catalogo. Para gerar embeddings das
solucoes, use:

```powershell
make solucoes-embedding
```

As solucoes usam a colecao Qdrant `bbsia_solucoes`. Documentos usam
`bbsia_chunks`.

## Upload, quarentena e aprovacao

`POST /upload` coloca PDFs em quarentena. A API valida:

- extensao `.pdf`;
- assinatura PDF;
- tamanho maximo;
- numero maximo de paginas;
- limite de caracteres extraidos;
- padroes simples de prompt injection.

Depois de revisar, aprove o arquivo:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/admin/quarantine/NOME_DO_ARQUIVO.pdf/approve `
  -Headers @{ "X-API-Key" = $env:ADMIN_API_KEY }
```

Reprocesse a base para indexar o arquivo aprovado.

## Rodar testes

Execute:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Ou use:

```powershell
make test
make lint
make typecheck
```

## Rodar benchmarks

Gere um dataset simples a partir de chunks:

```powershell
.\.venv\Scripts\python.exe -m bbsia.evaluation.benchmarks.generate_eval
```

Execute o benchmark RAG heuristico:

```powershell
.\.venv\Scripts\python.exe -m bbsia.evaluation.benchmarks.rag_benchmark
```

O resultado padrao fica em:

```text
bbsia/evaluation/benchmarks/results/
```

## Troubleshooting

### `Modelo indisponivel`

O modelo de embeddings ou reranker nao esta no cache local. Baixe o modelo,
desative o componente opcional ou ajuste `HF_LOCAL_FILES_ONLY`.

### `OLLAMA_URL deve apontar para loopback/local`

`ALLOW_REMOTE_OLLAMA=false` bloqueia URLs remotas. Use `localhost`,
`127.0.0.1` ou habilite explicitamente `ALLOW_REMOTE_OLLAMA=true`.

### `Dimensao inesperada do modelo de embeddings`

`EMBEDDING_DIM` nao corresponde ao modelo configurado. Ajuste a dimensao e
recrie o indice com `/reprocessar`.

### A API responde sem evidencia suficiente

Revise:

- se o indice foi gerado;
- se `GET /rag/health?load=true` mostra chunks carregados;
- se `MIN_DENSE_SCORE_PERCENT` esta alto demais;
- se filtros de area ou assunto eliminam os resultados esperados.

### Upload fica em quarentena

Esse comportamento e esperado. Revise `GET /admin/quarantine`, aprove o arquivo
e rode `/reprocessar`.
