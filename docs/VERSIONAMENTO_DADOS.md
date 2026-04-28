# Política de Versionamento de Dados

Esta política define quando manter arquivos de dados no Git, quando usar Git LFS e quando mover para dataset externo.

## Decisão atual (abril/2026)

- `docs/` permanece no Git tradicional.
- Motivos:
  - volume atual baixo (aprox. 5.28 MB no total);
  - arquivos individuais pequenos (maior ~2.33 MB);
  - baixo churn histórico em `docs/`.

## Regras de decisão

Use **Git tradicional** quando:
- arquivos de referência forem pequenos e estáveis;
- o total do diretório permanecer baixo;
- for importante clonar e rodar local sem etapa extra de download.

Use **Git LFS** quando:
- houver binários maiores e versões frequentes;
- arquivos individuais passarem de ~25 MB (ou limite interno do time);
- o histórico começar a crescer por revisões de PDFs/planilhas.

Use **dataset externo** (storage/datalake/release artifact) quando:
- houver dados sensíveis, licenciados ou volumosos;
- o conjunto for grande e com atualização frequente;
- fizer sentido separar código de dados e versionar por snapshot.

## Diretriz prática para este repositório

- `data/` e `uploads/` continuam fora do Git (já ignorados).
- `docs/` segue no Git até ultrapassar os limites acima.
- Ao cruzar os limites, abrir PR de migração para Git LFS ou dataset externo com:
  - inventário de arquivos;
  - impacto no onboarding/CI;
  - estratégia de rollback.

