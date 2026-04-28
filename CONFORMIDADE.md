# Conformidade do Piloto BBSIA

Este documento registra a revisao inicial de conformidade para o catalogo piloto e para o chatbot RAG local.

## Escopo

- Catalogo estruturado em `catalogo/solucoes_piloto.json`.
- Schema de validacao em `schemas/solucao_piloto.schema.json`.
- Pipeline local de embeddings e busca semantica.
- Uploads de PDF e reprocessamento local.

## Checklist

| Tema | Status | Evidencia | Acao recomendada |
|------|--------|-----------|------------------|
| LGPD - dados pessoais | Parcial | O schema exige `conformidade.dados_pessoais` e `base_legal_lgpd`; uploads passam por quarentena. | Confirmar se cada solucao processa dados pessoais reais antes de producao. |
| LGPD - minimizacao | Parcial | O catalogo piloto usa descricao operacional, sem campos pessoais obrigatorios. | Adicionar avaliacao de minimizacao por solucao quando houver dados reais. |
| LGPD - transparencia | Parcial | Respostas RAG incluem fontes/citacoes; catalogo tem descricao e orgao responsavel. | Criar texto de aviso ao usuario final se o chatbot for exposto fora do time. |
| Soberania de dados | Atendido no piloto local | `ALLOW_REMOTE_OLLAMA=false`, `HF_LOCAL_FILES_ONLY=true` e hospedagem local no schema. | Para nuvem, registrar regiao, operador, contratos e fluxo de dados. |
| Modelos locais | Parcial | Ollama local e embeddings Hugging Face locais sao configuraveis no `.env`. | Registrar versao, origem e licenca de cada modelo usado em ambiente oficial. |
| Open-source | Parcial | O schema exige `licenca_modelo` e `dependencias_open_source`; dependencias estao em `requirements.txt`. | Revisar licencas das dependencias e do modelo antes de publicacao externa. |
| Auditoria | Parcial | API grava eventos em `data/audit.log`; reprocessamento registra etapas. | Definir retencao, mascaramento e politica de acesso aos logs. |
| Uploads | Parcial | PDFs entram em quarentena e podem ser aprovados antes de indexacao. | Criar rotina operacional para revisao manual e exclusao de arquivos rejeitados. |

## Campos de conformidade por solucao

Cada item do catalogo piloto deve declarar:

- `dados_pessoais`: `nao`, `sim` ou `a_confirmar`.
- `base_legal_lgpd`: justificativa ou observacao de pendencia.
- `hospedagem`: `local`, `nuvem-brasil`, `nuvem-exterior`, `hibrida` ou `a_confirmar`.
- `modelo`: modelo principal usado pela solucao.
- `licenca_modelo`: licenca ou observacao de validacao pendente.
- `dependencias_open_source`: bibliotecas principais a revisar.

## Criterio para producao

Antes de sair de piloto, cada solucao deve ter:

- `dados_pessoais` diferente de `a_confirmar`;
- base legal LGPD revisada por responsavel do projeto;
- hospedagem definida e coerente com requisitos de soberania;
- licenca do modelo confirmada;
- dependencias open-source revisadas;
- evidencia de teste de recuperacao/qualidade documentada.
