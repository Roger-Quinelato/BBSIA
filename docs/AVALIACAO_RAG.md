# Avaliacao RAG

Este documento registra evidencias de qualidade do retrieval e decisoes de
calibracao para preservar a estabilidade das Fases 1 e 2 antes da Fase 3.

## Fechamento da Fase 2

A Fase 2 termina com o runtime RAG estabilizado, testes verdes e calibracao
inicial registrada. A calibracao atual nao autoriza mudanca de threshold, pois
os scores dense ficaram zerados na amostra executada. Assim, o fechamento da
Fase 2 nao deve tentar otimizar ranking sem uma evidencia melhor.

O trabalho de benchmark expandido passa a ser o primeiro gate da Fase 3:

- criar dataset v2 com perguntas in-scope e out-of-scope;
- medir retrieval, geracao, grounding e latencia;
- salvar baseline em `benchmarks/results/`;
- comparar qualquer experimento futuro contra esse baseline.
- impedir troca estrutural do RAG antes desse baseline.

Plano operacional: `docs/PLANO_BENCHMARK_RAG_FASE3.md`.

## Calibracao de threshold dense

- Data da calibracao: 2026-04-29
- Script: `scripts/calibrar_threshold.py`
- Evidencia principal: `benchmarks/results/threshold_calibration_latest.json`
- Copia de compatibilidade: `scripts/resultados_calibracao.json`
- Dataset/queries: 15 queries representativas em `QUERIES_DE_TESTE`, cobrindo
  areas `ia`, `infraestrutura`, `juridico`, `tecnologia`, `saude` e uma query
  fora de escopo.
- Top-k: 5
- Valor atual: `MIN_DENSE_SCORE_PERCENT=18`
- Valor sugerido pelo calculo bruto: `1`
- Valor adotado: `18`

### Justificativa

A execucao de 2026-04-29 retornou `score_dense=0.0000` para todas as queries,
incluindo as 14 queries dentro do escopo e a query fora do escopo. Assim, a
sugestao bruta de `1` nao representa uma melhoria real de separacao entre
consultas in-scope e out-of-scope; ela e um artefato de uma amostra sem sinal
dense informativo.

Por esse motivo, o valor `18` foi mantido. A decisao evita alterar o
comportamento padrao do retrieval sem evidencia de ganho. A proxima calibracao
deve primeiro confirmar que o caminho dense/Qdrant esta produzindo scores
positivos na evidencia antes de propor mudanca de threshold.

### Resultado observado

- Total de queries: 15
- Queries com resultado: 15
- In-scope: 14
- Out-of-scope: 1
- `dense_min`: 0.0000
- `dense_mediana`: 0.0000
- `dense_max`: 0.0000
- `out_scope_dense_max`: 0.0000
- Qualidade top-k por area/documento: 8/15 (53%)

### Criterio para ajuste futuro

Alterar `MIN_DENSE_SCORE_PERCENT` somente quando uma nova evidencia mostrar:

1. scores dense positivos para queries in-scope;
2. separacao mensuravel entre in-scope e out-of-scope;
3. pass rate igual ou superior ao valor atual em queries representativas;
4. decisao documentada neste arquivo ou em evidencia equivalente.

## Proximo baseline

A proxima avaliacao deve usar `benchmarks/datasets/rag_eval_dataset_v2.jsonl`,
conforme `docs/PLANO_BENCHMARK_RAG_FASE3.md`. Esse novo baseline sera o ponto
de comparacao para mudancas de Fase 3, incluindo query planning, variantes de
chunking, LangChain/RAGAS opcional, ajustes de reranker e qualquer alteracao de
threshold.

LangChain, RAGAS e DeepEval sao avaliadores opcionais e nao passam a ser
dependencia de runtime. Redis, Docker, Helm, OIDC/RBAC, auditoria em banco,
Elasticsearch, grafo e fine-tuning devem ser tratados como Fase 3 ou posterior,
nunca como pendencia retroativa da Fase 2.
