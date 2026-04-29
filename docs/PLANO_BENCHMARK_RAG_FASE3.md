# Plano de Benchmark RAG da Fase 3

Este plano define o gate de benchmark que deve abrir a Fase 3 do BBSIA. Ele
parte da arquitetura consolidada nas Fases 1 e 2 e cria um baseline
reproduzivel antes de qualquer troca estrutural, reorganizacao de pacotes ou
experimento com LangChain/RAGAS.

## Validade do plano

O plano e valido com estes enquadramentos:

- A arquitetura atual e o baseline oficial: Qdrant local, BM25 local, RRF,
  reranker opcional, Ollama local e faithfulness no pipeline.
- A calibracao inicial ja esta registrada e mantem
  `MIN_DENSE_SCORE_PERCENT=18` por decisao documentada.
- O benchmark nao deve alterar `api.py`, `pipeline.py`, `retriever.py`,
  `embedding.py` ou os artefatos de dados gerados.
- LangChain, RAGAS e DeepEval entram apenas como camada opcional de avaliacao,
  nunca como dependencia obrigatoria do runtime.
- A calibracao de `MIN_DENSE_SCORE_PERCENT` continua sendo evidencia, nao
  mudanca automatica de configuracao.
- Nenhuma troca estrutural deve ocorrer antes de salvar o baseline em
  `benchmarks/results/`.

O unico ajuste em relacao ao plano externo e tratar este trabalho como primeiro
marco da Fase 3, nao como pendencia bloqueante da Fase 2. A Fase 2 termina com
runtime estavel e evidencia inicial de calibracao; a Fase 3 comeca medindo o
baseline de qualidade com dataset expandido.

## Objetivo

Medir a qualidade real do RAG atual antes de iniciar mudancas estruturais da
Fase 3, permitindo comparacao futura contra:

- query planning mais forte;
- splitters alternativos;
- variantes LangChain;
- modelos de embedding ou LLM diferentes;
- ajustes de threshold, reranker ou chunking.

Smoke tests provam que o sistema liga. Benchmark prova se recupera o contexto
certo, responde com fidelidade e rejeita perguntas fora de escopo.

## Escopo

### Incluido

- Dataset expandido de avaliacao.
- Benchmark heuristico oficial sem dependencias pesadas.
- Evidencia de threshold e out-of-scope.
- Medicao basica de latencia.
- Opcionalmente, avaliacao RAGAS/DeepEval em ambiente separado.

### Fora de escopo

- Trocar backend vetorial.
- Trocar chunking.
- Substituir retrieval por LangChain.
- Mudar comportamento padrao de `search()`.
- Tornar RAGAS/LangChain dependencia obrigatoria.
- Introduzir Redis, Docker, Helm, OIDC/RBAC, auditoria em banco,
  Elasticsearch, grafo ou fine-tuning como requisito deste gate. Esses temas
  pertencem a Fase 3 posterior ou etapas seguintes, depois do baseline.

## Artefatos atuais

- `benchmarks/rag_benchmark.py`: benchmark leve oficial, com metricas
  heuristicas e fallback opcional para DeepEval quando instalado.
- `benchmarks/datasets/rag_eval_dataset.jsonl`: dataset inicial com poucos
  casos.
- `benchmarks/benchmark.py`: esqueleto opcional RAGAS/LangChain.
- `benchmarks/eval_dataset.json`: dataset RAGAS ainda vazio.
- `scripts/calibrar_threshold.py`: calibracao com 15 queries e evidencia em
  `benchmarks/results/threshold_calibration_latest.json`.
- `docs/AVALIACAO_RAG.md`: decisao atual de manter
  `MIN_DENSE_SCORE_PERCENT=18`, pois a ultima calibracao retornou scores dense
  zerados e nao acionaveis.

### Lacuna planejada para o primeiro prompt da Fase 3

O `benchmarks/rag_benchmark.py` atual ja mede `context_recall`,
`faithfulness_score` e `answer_relevancy` sem exigir dependencias pesadas. Para
abrir a Fase 3, ele deve ser expandido para consumir o dataset v2 e registrar,
sem mudar `search()` nem `answer_question()`:

- Hit@k por `expected_document_fragment`;
- out-of-scope rejection por campo `escopo`;
- `score_dense` top-1 e outros sinais de retrieval ja retornados pelo runtime;
- latencia de retrieval, geracao e total end-to-end;
- resumo Go/No-Go salvo junto dos resultados em `benchmarks/results/`.

Essa expansao pertence ao gate inicial da Fase 3. Ela nao e pendencia
retroativa da Fase 2 e nao deve introduzir LangChain, RAGAS ou DeepEval como
dependencia obrigatoria.

## Dataset v2

Criar `benchmarks/datasets/rag_eval_dataset_v2.jsonl` com 28 perguntas, uma por
linha em JSON.

Campos esperados:

```json
{
  "id": "q01",
  "categoria": "bbsia_mvp",
  "question": "...",
  "reference_answer": "...",
  "expected_context_terms": ["termo1", "termo2"],
  "expected_document_fragment": "nome-parcial-do-arquivo",
  "filtro_area": ["ia"],
  "filtro_assunto": [],
  "escopo": "in"
}
```

Distribuicao sugerida:

- q01-q05: BBSIA, MVP e fases do projeto.
- q06-q10: RAG, chatbot e pipeline tecnico.
- q11-q14: infraestrutura.
- q15-q19: etica, LGPD e conformidade.
- q20-q23: prontidao em IA e maturidade.
- q24-q28: perguntas fora de escopo.

As queries de `scripts/calibrar_threshold.py` podem ser reutilizadas como ponto
de partida, mas o dataset v2 deve conter tambem documento esperado, termos
esperados e marcador de escopo.

## Metricas por camada

### Retrieval

- Hit@k: documento esperado aparece no top-k.
- Context Term Recall: proporcao de termos esperados presentes nos contextos.
- Dense Score Top-1: `score_dense` do primeiro resultado.
- Out-of-scope Rejection: perguntas fora de escopo devem cair abaixo do
  threshold ou acionar resposta sem evidencia.
- Reranker Delta: melhoria ou neutralidade do reranker em relacao ao ranking
  bruto, quando medido.

Metas iniciais:

- Hit@5 >= 85% para perguntas in-scope.
- Context Term Recall medio >= 0.75.
- Out-of-scope rejection >= 90%, idealmente 100%.

### Geracao

- Response relevancy heuristica.
- Presenca de citacao quando ha evidencia.
- No-evidence rate baixo para perguntas in-scope.
- No-evidence correto para perguntas out-of-scope.
- Error rate igual a 0%.

### Grounding

- Token overlap faithfulness.
- Resultado do NLI gate interno quando aplicavel.
- Opcionalmente, RAGAS `faithfulness`, `answer_relevancy`,
  `context_recall` e `context_precision`.

### Operacao

- Latencia de `retriever.search()`.
- Latencia do reranker.
- Latencia de geracao.
- Latencia total end-to-end.
- Timeouts Ollama.
- Memoria durante preload, quando medido.

## Checklist de entrada da Fase 3

1. Rodar a suite completa:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

2. Criar `benchmarks/datasets/rag_eval_dataset_v2.jsonl`.

3. Expandir `benchmarks/rag_benchmark.py` para ler os campos do dataset v2
   (`expected_document_fragment`, `escopo`, filtros e termos esperados) e medir
   retrieval, geracao, grounding, out-of-scope e latencia sem alterar
   `search()` nem `answer_question()`.

4. Expandir ou parametrizar `scripts/calibrar_threshold.py` para usar o dataset
   v2 sem remover a compatibilidade atual.

5. Rodar calibracao:

```powershell
.\.venv\Scripts\python.exe scripts\calibrar_threshold.py
```

6. Rodar benchmark heuristico:

```powershell
.\.venv\Scripts\python.exe benchmarks\rag_benchmark.py `
  --dataset benchmarks\datasets\rag_eval_dataset_v2.jsonl `
  --output benchmarks\results\rag_benchmark_baseline.json
```

7. Criar `benchmarks/results/baseline_summary.md` com:

- data;
- modelo Ollama;
- embedding model;
- reranker model;
- `TOP_K`;
- `MIN_DENSE_SCORE_PERCENT`;
- metricas por camada;
- decisao de Go/No-Go para experimentos estruturais.

8. Opcionalmente, instalar extras de avaliacao em ambiente separado e rodar
   RAGAS/DeepEval. Esses pacotes nao devem entrar no runtime obrigatorio.

## Go/No-Go

- Hit@5 >= 85% e faithfulness >= 0.70: Fase 3 pode focar em features novas e
  modularizacao.
- Hit@5 < 70%: investigar retrieval, filtros, reranker ou chunking antes de
  trocar arquitetura.
- Faithfulness < 0.50: revisar prompt, contexto, modelo ou fallback.
- Out-of-scope rejection < 90%: revisar threshold e no-evidence gate.
- Latencia E2E > 60 s: perfilar busca, reranker e geracao antes de expandir.

## Relacao com a modularizacao

Este benchmark deve vir antes de qualquer mudanca grande prevista em
`docs/PLANO_MODULARIZACAO_FASE3.md`. A modularizacao so deve prosseguir depois
que o baseline estiver salvo, porque ele sera o comparador para saber se uma
mudanca futura melhorou ou degradou o RAG.
