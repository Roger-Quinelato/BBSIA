# TASKS.md: Plano de Execução (Spec-Driven Development)

Este documento descreve o plano de execução para implementar o RAG Semântico para Solução de Problemas.

## 1. Mapa de Dependências e Ordem de Execução

- **Fase 1: Fundação do Domínio (Paralelizável)**
  - `[TASK-1.1]` Expandir o Schema de Soluções
  - `[TASK-1.2]` Criar o Catálogo Mock de Soluções (Depende de 1.1)
  - `[TASK-1.3]` Corrigir a Ingestão e Vetorização de Soluções (Depende de 1.2)
- **Fase 2: Motor de Recuperação (Sequencial após Fase 1)**
  - `[TASK-2.1]` Adaptar o Retriever para Múltiplas Coleções (Depende de 1.3)
- **Fase 3: Orquestração e Prompting (Sequencial após Fase 2)**
  - `[TASK-3.1]` Refatorar o Pipeline para Roteamento de Diagnóstico (Depende de 2.1)
  - `[TASK-3.2]` Ajustar o Generator e Prompts (Depende de 3.1)
- **Fase 4: Validação (Final)**
  - `[TASK-4.1]` Atualizar Benchmarks e Testes Unitários (Depende de 3.2)

---

## 2. Detalhamento das Tarefas

### [TASK-1.1] Expandir o Schema de Soluções
- **Onde será feito:** `schemas/solucao_piloto.schema.json` e as classes Pydantic associadas em `api_core.py`.
- **O que será reutilizado:** O esquema base atual de "solução piloto".
- **Pré-requisitos:** Nenhum.
- **Critérios de Aceite (DoD):**
  - O JSON schema inclui propriedades obrigatórias: `sintomas`, `causa_raiz`, `pre_condicoes`, `passos_implantacao`, `riscos`.
  - As classes Pydantic refletem essas mudanças e os testes em `tests/test_catalogo_solucoes.py` passam.

### [TASK-1.2] Criar o Catálogo Inicial de Soluções
- **Onde será feito:** `catalogo/solucoes_piloto.json` (novo arquivo).
- **O que será reutilizado:** Conhecimento de negócios/exemplos da documentação existente.
- **Pré-requisitos:** `[TASK-1.1]`.
- **Critérios de Aceite (DoD):**
  - O diretório `catalogo/` existe e contém pelo menos 3 exemplos completos de soluções estruturadas válidas perante o novo schema Pydantic.

### [TASK-1.3] Corrigir a Ingestão e Vetorização de Soluções
- **Onde será feito:** `embedding.py`, `scripts/gerar_embeddings_solucoes.py`, `vector_store.py`.
- **O que será reutilizado:** Lógica do `run_embedding`, modelos E5.
- **Pré-requisitos:** `[TASK-1.2]`.
- **Critérios de Aceite (DoD):**
  - Executar o script de ingestão de soluções escreve os vetores em uma coleção/namespace separado (ex: `bbsia_solucoes`) no Qdrant.
  - A coleção principal `bbsia_chunks` permanece inalterada após a ingestão das soluções.

### [TASK-2.1] Adaptar o Retriever para Múltiplas Coleções
- **Onde será feito:** `retriever.py`.
- **O que será reutilizado:** Lógica de RRF, busca Dense e BM25.
- **Pré-requisitos:** `[TASK-1.3]`.
- **Critérios de Aceite (DoD):**
  - O método `search()` aceita um parâmetro `target_collection` (ou similar) para buscar apenas em documentos, apenas em soluções, ou em ambos separadamente.
  - Testes unitários do retriever validam buscas em coleções distintas sem conflito.

### [TASK-3.1] Refatorar o Pipeline para Roteamento de Diagnóstico
- **Onde será feito:** `pipeline.py`.
- **O que será reutilizado:** A fachada de `answer_question` ou métodos do pipeline principal.
- **Pré-requisitos:** `[TASK-2.1]`.
- **Critérios de Aceite (DoD):**
  - O pipeline avalia a query. Se for identificada como uma busca por problema/solução, aciona o fluxo de diagnóstico (busca no catálogo de soluções e em documentos de suporte).
  - O contexto (context string) passado para o LLM é formatado diferenciando as "Soluções Candidatas" das "Evidências Documentais".

### [TASK-3.2] Ajustar o Generator e Prompts
- **Onde será feito:** `generator.py` (e possivelmente `routers/rag.py`).
- **O que será reutilizado:** Endpoint do Ollama.
- **Pré-requisitos:** `[TASK-3.1]`.
- **Critérios de Aceite (DoD):**
  - O prompt injetado no Ollama obriga que a saída siga uma estrutura clara (Diagnóstico, Solução Recomendada, Passos, Riscos) baseada no contexto.
  - O LLM não gera soluções inventadas (hallucination); se não houver solução no catálogo, responde com um fallback conservador.

### [TASK-4.1] Atualizar Benchmarks e Testes Unitários
- **Onde será feito:** `benchmarks/eval_dataset.json`, `benchmarks/rag_benchmark.py`, pasta `tests/`.
- **O que será reutilizado:** Scripts de benchmark existentes.
- **Pré-requisitos:** `[TASK-3.2]`.
- **Critérios de Aceite (DoD):**
  - Pelo menos 5 queries de problemas são adicionadas ao benchmark com respostas esperadas mapeadas para as soluções do catálogo.
  - A suíte de testes (`pytest`) passa com 100% de sucesso sem quebrar os testes anteriores do RAG documental.
