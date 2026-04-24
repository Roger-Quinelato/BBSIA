# Plano: Metadados Inteligentes de Artigos Científicos → Frontend

## Visão geral do fluxo

```
┌─────────────────────────────────────────────────────────────────────┐
│                           HOJE                                       │
│  PDF → extração de texto → chunks (metadados manuais) → FAISS       │
│  Frontend: áreas hardcoded, sem contexto das fontes                  │
└─────────────────────────────────────────────────────────────────────┘

                               ↓  este plano

┌─────────────────────────────────────────────────────────────────────┐
│                          OBJETIVO                                    │
│  PDF → classificador inteligente → metadados JSON                   │
│      → chunks enriquecidos → FAISS                                  │
│      → biblioteca.json (catálogo persistente)                       │
│      → API expõe dinamicamente                                      │
│      → Frontend: filtros reais, biblioteca de fontes, citações ricas│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Fase 1 — Schema de metadados e extrator inteligente

**Novo arquivo:** `src/classificador_artigo.py`

### 1.1 Schema do metadado de documento

```json
{
  "id": "2022_MarinadeAlencarAraripeCoutinho",
  "titulo": "Adoção de Inteligência Artificial em Serviços Públicos Brasileiros",
  "autores": ["Marina de Alencar Araripe Coutinho"],
  "ano": 2022,
  "instituicao": "LIIA / Universidade de Brasília",
  "tipo_documento": "artigo_cientifico",
  "resumo": "Este artigo analisa a adoção de IA...",
  "palavras_chave": ["inteligência artificial", "setor público", "LGPD", "ética"],
  "area_tematica": "ia",
  "assuntos": ["etica", "governanca", "transformacao-digital", "lgpd"],
  "metodologia": "estudo de caso",
  "secoes_detectadas": ["Introdução", "Referencial Teórico", "Metodologia", "Resultados", "Conclusão"],
  "paginas_total": 24,
  "documento_original": "docs/2022_MarinadeAlencarAraripeCoutinho.pdf",
  "data_ingestao": "2026-04-24T12:30:00Z",
  "qualidade_extracao": "alta"
}
```

### 1.2 Estratégia de extração (duas camadas)

**Camada 1 — Heurísticas (rápida, sem LLM):**
- **Título:** maior fonte da 1ª página (via `_span_lines()`)
- **Autores:** padrões regex na 1ª e 2ª página (`"Nome Sobrenome¹"`, `"Nome Sobrenome, Cargo"`)
- **Ano:** regex `(19|20)\d{2}` em metadados do PDF ou primeiras páginas
- **Resumo:** busca pelo bloco após palavras-chave `"Resumo"`, `"Abstract"`, `"Summary"`
- **Seções:** extração das seções já detectadas pelo `_extract_text_elements()`
- **Tipo de documento:** heurística por padrões (seções como "Metodologia" → artigo; ausência → relatório/manual)

**Camada 2 — Classificação por LLM (enriquecimento):**

Enviar o título + resumo extraídos para o Ollama com o seguinte prompt:

```
Você receberá o título e resumo de um documento.
Responda APENAS com um JSON válido no seguinte formato:
{
  "area_tematica": "<ia|saude|infraestrutura|juridico|tecnologia|geral>",
  "assuntos": ["<lista de 3 a 6 tags em português, minúsculas, sem acentos>"],
  "palavras_chave": ["<3 a 8 palavras-chave do próprio texto>"],
  "metodologia": "<estudo de caso|revisao-sistematica|pesquisa-quantitativa|relatorio|manual|outro>",
  "tipo_documento": "<artigo_cientifico|relatorio_tecnico|manual|apresentacao|outro>"
}
```

**Fallback:** se o LLM falhar ou retornar JSON inválido, usar os valores heurísticos e marcar `"qualidade_extracao": "baixa"`.

### 1.3 Novo arquivo: `data/biblioteca.json`

Catálogo persistente de todos os documentos ingeridos:

```json
{
  "versao": 1,
  "atualizado_em": "2026-04-24T12:30:00Z",
  "documentos": [
    { ...metadado do doc 1... },
    { ...metadado do doc 2... }
  ]
}
```

---

## Fase 2 — Integração com o pipeline existente

### 2.1 `extrator_pdf_v2.py`

Adicionar ao campo de cada documento no `documentos_extraidos_v2.json`:

```json
{
  "documento": "2022_MarinadeAlencarAraripeCoutinho.pdf",
  "metadados": { ...schema da fase 1... },
  "paginas": [ ...extração atual... ]
}
```

A função `run_extraction()` ficará responsável por:
1. Extrair o texto normalmente
2. Chamar `classificador_artigo.classificar(pdf_path, pages)` 
3. Salvar o metadado no JSON estruturado E atualizar o `data/biblioteca.json`

### 2.2 `chunking.py`

**Substituir** o mapeamento manual `CATEGORIAS_DOCUMENTOS` pelos metadados vindos do JSON:

```python
# ANTES (hoje):
CATEGORIAS_DOCUMENTOS = {
    "Bases para Chatbot BBSIA e RAG.pdf": {"area": "ia", ...}
}

# DEPOIS:
def get_doc_metadata(filepath: str) -> dict:
    # 1. Tenta upload_metadata (mais específico, vem do usuário)
    # 2. Tenta biblioteca.json (metadados classificados automaticamente)  ← NOVO
    # 3. Fallback genérico {"area": "geral", "assuntos": ["geral"]}
```

**Enriquecer cada chunk** com campos de autoria:

```json
{
  "id": 42,
  "texto": "...",
  "area": "ia",
  "assuntos": ["etica", "lgpd"],
  "documento": "2022_MarinadeAlencarAraripeCoutinho.pdf",
  "doc_titulo": "Adoção de IA em Serviços Públicos",
  "doc_autores": ["Marina de Alencar Araripe Coutinho"],
  "doc_ano": 2022,
  "pagina": 14
}
```

### 2.3 `rag_engine.py`

Atualizar a formatação das citações inline no `PROMPT_TEMPLATE`:

```
# ANTES:
[Fonte 1] LIIA BBSIA - Infra-estrutura.pdf (p. 3)

# DEPOIS:
[Fonte 1] Coutinho, 2022 — "Adoção de IA em Serviços Públicos" (p. 14)
```

---

## Fase 3 — Novos endpoints da API

### `GET /biblioteca`

Retorna o catálogo completo de documentos com metadados:

```json
{
  "total": 7,
  "documentos": [
    {
      "id": "2022_MarinadeAlencarAraripeCoutinho",
      "titulo": "Adoção de IA em Serviços Públicos Brasileiros",
      "autores": ["Marina de Alencar Araripe Coutinho"],
      "ano": 2022,
      "area_tematica": "ia",
      "assuntos": ["etica", "lgpd"],
      "tipo_documento": "artigo_cientifico",
      "paginas_total": 24
    }
  ]
}
```

Suporta filtros via query string:
- `GET /biblioteca?area=ia`
- `GET /biblioteca?tipo=artigo_cientifico`
- `GET /biblioteca?ano_min=2020&ano_max=2023`

### `GET /biblioteca/{doc_id}`

Retorna metadado completo de um documento específico (incluindo resumo e seções detectadas).

### `GET /filtros`

Retorna os valores únicos disponíveis para cada filtro, para popular os dropdowns do frontend dinamicamente:

```json
{
  "areas": ["ia", "infraestrutura", "juridico"],
  "tipos": ["artigo_cientifico", "relatorio_tecnico"],
  "anos": [2022, 2023, 2026],
  "assuntos": ["etica", "lgpd", "rag", "chatbot", "kubernetes"]
}
```

---

## Fase 4 — Frontend (web/index.html + web/app.js)

### 4.1 Painel "Biblioteca" na sidebar

Substituir a lista estática de agentes por um painel dinâmico carregado da API:

```
┌─────────────────────────┐
│ 📚 Biblioteca (7)       │
├─────────────────────────┤
│ 🔬 Coutinho, 2022       │
│    Adoção de IA em...   │
│    [ia] [ética] [lgpd]  │
├─────────────────────────┤
│ 📋 LIIA BBSIA - MVP     │
│    Fase 1 do Banco...   │
│    [ia] [rag] [mvp]     │
├─────────────────────────┤
│ 📊 Infra-estrutura      │
│    Requisitos técnicos  │
│    [infra] [kubernetes] │
└─────────────────────────┘
```

Clicar em um documento aplica o filtro automaticamente na próxima consulta.

### 4.2 Filtros dinâmicos

Substituir as áreas hardcoded no `app.js` por chamada à API `GET /filtros`:

```javascript
// ANTES (hardcoded):
const GENERAL_AREAS = ["tecnologia", "ia", "saude", "infraestrutura", "juridico"];

// DEPOIS (dinâmico):
async function loadFilters() {
  const { areas, tipos, anos } = await fetch("/filtros").then(r => r.json());
  // popula os selects dinamicamente
}
```

### 4.3 Citações ricas nas respostas

Quando a API retornar chunks com metadados de autoria, exibir as fontes com formato acadêmico:

```
┌─────────────────────────────────────────────────────┐
│ BBSIA:                                               │
│ A adoção de IA no setor público requer frameworks... │
│                                                      │
│ 📎 Fontes:                                           │
│  [1] Coutinho (2022) — Adoção de IA em Serviços...  │
│      p. 14 | artigo científico | área: ia            │
│  [2] LIIA BBSIA - Fase 1 MVP (2026) — p. 7          │
│      relatório técnico | área: ia                    │
└─────────────────────────────────────────────────────┘
```

### 4.4 Tela de detalhes do documento (modal)

Clicar em uma fonte exibe um modal com:
- Título completo, autores, ano, instituição
- Resumo
- Palavras-chave como tags
- Link para abrir o PDF original (se disponível)

---

## Fase 5 — Testes e validação

### Novos testes a criar

| Arquivo | Teste |
|---------|-------|
| `tests/test_classificador.py` | Extração de título, autores e ano da 1ª página |
| `tests/test_classificador.py` | Fallback quando LLM retorna JSON inválido |
| `tests/test_classificador.py` | Detecção de tipo de documento (artigo vs relatório) |
| `tests/test_api.py` | `GET /biblioteca` retorna lista de documentos |
| `tests/test_api.py` | `GET /filtros` retorna áreas únicas |
| `tests/test_api.py` | Chunks contêm `doc_titulo` e `doc_autores` |

---

## Dependências novas

| Pacote | Motivo | Obrigatório? |
|--------|--------|-------------|
| nenhum | Heurísticas usam PyMuPDF e regex já instalados | — |
| `pydantic` | Já incluído via FastAPI | ✅ já existe |

> [!IMPORTANT]
> A classificação por LLM usa o Ollama que já está no projeto. **Nenhuma dependência nova** é necessária para as fases 1–4.

---

## Ordem de implementação recomendada

```
Etapa 1 ──► Schema Pydantic do metadado (15 min)
Etapa 2 ──► Extração heurística no classificador_artigo.py (2h)
Etapa 3 ──► Classificação LLM + fallback (1h)
Etapa 4 ──► Integração com extrator_pdf_v2.py + biblioteca.json (1h)
Etapa 5 ──► Enriquecimento dos chunks no chunking.py (30 min)
Etapa 6 ──► Citações ricas no rag_engine.py (30 min)
Etapa 7 ──► Endpoints GET /biblioteca e GET /filtros na api.py (1h)
Etapa 8 ──► Painel Biblioteca e filtros dinâmicos no frontend (2h)
Etapa 9 ──► Testes (1h)
```

**Tempo estimado total: ~9 horas de desenvolvimento**

---

## Impacto para o usuário final

| Antes | Depois |
|-------|--------|
| Filtros fixos: ia, saúde, infra... | Filtros reais dos documentos na base |
| "Fonte: arquivo.pdf (p. 3)" | "Coutinho, 2022 — p. 14" |
| Sidebar com áreas genéricas | Biblioteca navegável de documentos |
| Upload sem contexto | Upload classifica o artigo automaticamente |
| Usuário não sabe o que está na base | Painel mostra cada documento com resumo e tags |
