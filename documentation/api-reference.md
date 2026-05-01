# Referencia da API

Esta pagina documenta os endpoints implementados pela API FastAPI em
`bbsia/app/routers/`.

## Autenticacao

A API usa o header `X-API-Key` quando voce configura uma chave no `.env`.

- Endpoints de leitura aceitam `READ_API_KEY` ou `ADMIN_API_KEY`.
- Endpoints administrativos aceitam apenas `ADMIN_API_KEY`.
- Se as chaves ficam vazias, o middleware nao exige autenticacao.
- `GET /`, `GET /status`, `GET /docs`, `GET /redoc`, `GET /openapi.json` e
  arquivos em `/web` sao publicos.

O middleware tambem aplica rate limit por IP com `RATE_LIMIT_REQUESTS` e
`RATE_LIMIT_WINDOW_SEC`.

## Sistema

### `GET /`

Retorna o status basico da API. Se a pasta web existir, redireciona para
`/web`.

### `GET /status`

Retorna a saude geral da aplicacao.

Resposta inclui:

- status da API;
- estado do indice RAG;
- total de chunks;
- quantidade de areas e assuntos;
- status do Ollama;
- modelos disponiveis;
- versao da API;
- snapshot do reprocessamento.

## RAG

### `POST /chat`

Conversa com o chatbot RAG. O endpoint retorna `application/x-ndjson` em
streaming.

Exemplo:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/chat `
  -ContentType "application/json" `
  -Body '{
    "pergunta": "Quais sao os principais riscos de um chatbot RAG?",
    "top_k": 5,
    "filtro_area": ["ia"],
    "filtro_assunto": ["rag"]
  }'
```

Campos do payload:

| Campo | Tipo | Obrigatorio | Descricao |
|---|---|---:|---|
| `pergunta` | string | Sim | Pergunta do usuario. |
| `modelo` | string | Nao | Modelo Ollama permitido. Usa `DEFAULT_MODEL`. |
| `top_k` | integer | Nao | Total de resultados. Valor entre 1 e 20. |
| `filtro_area` | array | Nao | Areas para filtrar resultados. |
| `filtro_assunto` | array | Nao | Assuntos para filtrar resultados. |
| `conversation_id` | string | Nao | Mantem historico curto da conversa. |

Eventos NDJSON:

- `metadata`: fontes, resultados, prompt e modo diagnostico.
- `token`: partes da resposta gerada.
- `faithfulness`: resultado opcional de checagem de fidelidade.
- `error`: erro durante o streaming.

### `POST /search`

Busca documentos no indice RAG sem gerar resposta pelo LLM.

Exemplo:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/search `
  -ContentType "application/json" `
  -Body '{
    "query": "governanca de IA no setor publico",
    "top_k": 5
  }'
```

Campos do payload:

| Campo | Tipo | Obrigatorio | Descricao |
|---|---|---:|---|
| `query` | string | Sim | Texto da busca. |
| `top_k` | integer | Nao | Total de resultados. Valor entre 1 e 20. |
| `filtro_area` | array | Nao | Areas para filtrar resultados. |
| `filtro_assunto` | array | Nao | Assuntos para filtrar resultados. |

### `GET /areas`

Lista areas disponiveis no indice carregado.

### `GET /assuntos`

Lista assuntos disponiveis no indice carregado.

### `GET /modelos`

Lista modelos Ollama permitidos e o modelo padrao.

### `GET /rag/health`

Retorna o estado do cache RAG.

Parametros:

| Parametro | Tipo | Padrao | Descricao |
|---|---|---|---|
| `load` | boolean | `false` | Carrega recursos se o cache estiver vazio. |

## Biblioteca

### `GET /biblioteca`

Lista documentos registrados na biblioteca.

Filtros opcionais:

| Parametro | Tipo | Descricao |
|---|---|---|
| `area` | string | Filtra por area tematica. |
| `tipo` | string | Filtra por tipo de documento. |
| `ano_min` | integer | Retorna documentos a partir do ano informado. |
| `ano_max` | integer | Retorna documentos ate o ano informado. |

### `GET /biblioteca/{doc_id}`

Retorna os metadados completos de um documento da biblioteca.

### `GET /filtros`

Lista valores disponiveis para areas, tipos, anos e assuntos.

## Upload e administracao

### `POST /upload`

Envia PDFs para quarentena. O endpoint valida extensao, assinatura PDF, tamanho,
paginas, texto extraido e sinais de prompt injection.

Exemplo:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/upload `
  -Headers @{ "X-API-Key" = $env:ADMIN_API_KEY } `
  -Form @{
    area = "ia"
    assuntos = "rag,governanca"
    files = Get-Item ".\docs\meu-documento.pdf"
  }
```

Arquivos aprovados entram em `uploads/approved` e podem participar do proximo
reprocessamento. Arquivos enviados entram primeiro em `uploads/quarantine`.

### `POST /upload-metadata`

Atualiza metadados de um documento enviado.

Payload:

```json
{
  "documento": "uploads/approved/meu-documento.pdf",
  "area": "ia",
  "assuntos": ["rag", "governanca"]
}
```

### `GET /admin/quarantine`

Lista arquivos em quarentena, seus metadados, status de validacao e achados de
prompt injection.

### `POST /admin/quarantine/{stored_filename}/approve`

Aprova um arquivo em quarentena e move o PDF para `uploads/approved`.

### `POST /reprocessar`

Enfileira o reprocessamento da base. A fila executa reprocessamentos em serie e
publica o estado em `GET /status`.

### `POST /recarregar`

Recarrega recursos RAG em memoria depois que o indice muda.

## Metricas

Quando `prometheus-fastapi-instrumentator` esta instalado, a API expoe metricas
Prometheus em:

```text
GET /metrics
```
