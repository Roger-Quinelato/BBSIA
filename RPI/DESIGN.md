# DESIGN.md: Decisões Arquiteturais e Reuso

## 1. Visão Geral da Arquitetura
A arquitetura transitará de um modelo RAG singular focado em "Knowledge Retrieval" para um modelo dual que incorpora o domínio estruturado de "Problem Resolution". Isso significa que o pipeline precisará consultar dois domínios de dados distintos (documentos não estruturados vs. catálogo de soluções estruturadas) e orquestrar a resposta para formatar um diagnóstico e uma recomendação.

## 2. Decisões Arquiteturais de Alto Nível (ADRs)

### ADR 01: Separação Lógica dos Índices de Vetores
- **Decisão:** Os chunks de documentos gerais e os chunks de soluções piloto não devem habitar a mesma coleção no banco de dados vetorial de forma indiscriminada.
- **Implementação:** O Qdrant local será configurado para gerenciar coleções separadas (ex: `bbsia_chunks` para documentos e `bbsia_solucoes` para o catálogo), ou, alternativamente, utilizar filtragem rígida de metadados se mantidos na mesma coleção, assegurando que o script de ingestão de soluções (`gerar_embeddings_solucoes.py`) não destrua ou sobrescreva o índice de documentos.

### ADR 02: Contrato Estruturado do Catálogo de Soluções
- **Decisão:** O conceito de "Solução Piloto" será o artefato de primeira classe para o diagnóstico.
- **Implementação:** O schema atual (`schemas/solucao_piloto.schema.json`) será expandido para incluir propriedades semânticas cruciais para o RAG: `sintomas_atendidos`, `causa_raiz`, `pre_condicoes`, `passos_implantacao`, `riscos` e `restricoes`. O arquivo mestre JSON do catálogo residirá em `catalogo/solucoes_piloto.json`.

### ADR 03: Pipeline de Orquestração Orientado a Diagnóstico
- **Decisão:** O fluxo de geração de resposta será enriquecido para detectar a intenção de resolução de problemas e ajustar o prompt dinamicamente.
- **Implementação:** O `pipeline.py` orquestrará a recuperação (Retrieval). Quando o usuário relatar um problema, o retriever buscará no índice de soluções as *guidelines* corretas e no índice de documentos as *evidências* ou contextos de apoio. O LLM será instruído a formatar a saída com as seções da solução recomendada.

## 3. Reuso de Componentes Existentes
Para evitar alucinações e retrabalho, as seguintes abstrações existentes **devem** ser reutilizadas como base:

- **Infraestrutura Vetorial (`vector_store.py`):** O client do Qdrant local será reutilizado. Novas funções para suportar múltiplas coleções devem ser adicionadas a este módulo em vez de criar conexões concorrentes.
- **Mecanismo de Recuperação (`retriever.py`):** A lógica de busca híbrida (Dense Embedding + Sparse BM25 + Reciprocal Rank Fusion) e o reranker serão mantidos. O método `search()` deverá ser parametrizado para indicar qual "coleção/índice" deve ser consultado.
- **Geração e LLM (`generator.py`):** A interface com a API do Ollama e o formato de payload permanecem os mesmos. Apenas o template do prompt deverá ser alterado no orquestrador antes de chamar o `generator.py`.
- **API Core (`api_core.py` e `routers/rag.py`):** A estrutura do FastAPI, esquemas base do Pydantic (`ChatRequest`, `ChatResponse`), middlewares de CORS e tratamento de erros serão preservados. O endpoint `/chat` continuará sendo a porta de entrada principal.
- **Embeddings (`embedding.py`):** A geração de embeddings E5 será reutilizada, sendo necessário apenas parametrizar o diretório de destino e nome da coleção para as soluções.
