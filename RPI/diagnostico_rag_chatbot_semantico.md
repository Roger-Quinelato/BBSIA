# Diagnostico investigativo: Chatbot com RAG semantico para solucoes de problemas com IA

Data da varredura: 2026-04-30

## Escopo da investigacao

- Objetivo analisado: evoluir a base para um chatbot com RAG semantico voltado a solucao de problemas usando IA.
- Nao houve escrita de codigo de implementacao. Este arquivo e o unico artefato criado.
- Nao usei subagentes porque a base relevante e pequena o suficiente para leitura direta sem risco de estouro de contexto.
- Validacao executada: `.\.venv\Scripts\python.exe -m pytest -q`.
- Resultado dos testes: 68 testes passaram. Houve apenas avisos de dependencias do Docling/RapidOCR e uma excecao de limpeza de diretorio temporario do pytest por permissao no Windows ao encerrar.

## Visao executiva

- A base ja possui um RAG funcional em Python com FastAPI, Qdrant local, SentenceTransformers, busca hibrida dense + BM25, RRF, reranker opcional, Ollama para geracao e endpoints REST/streaming.
- O fluxo atual esta mais maduro para perguntas sobre documentos indexados do que para recomendacao estruturada de solucoes de problemas.
- Existe um embriao de dominio de "solucoes piloto" em `catalogo_solucoes.py` e `schemas/solucao_piloto.schema.json`, mas ele nao esta integrado ao indice oficial usado por `/chat` e `/search`.
- A pasta esperada `catalogo/` nao existe na copia investigada, e o indice atual nao contem chunks com `content_type = solucao_piloto`.
- A arquitetura atual resolve "perguntar aos documentos"; para resolver "diagnosticar problema e recomendar solucao", faltam modelo de dominio, ingestao unificada de solucoes, filtros/metadata especificos, ranking orientado a problema e contrato de resposta.

## Fluxo de dados atual

- Entrada principal da API: `bbsia/app/main.py` cria o app FastAPI e registra routers de RAG, admin, sistema e biblioteca.
- App/middlewares/schemas: `bbsia/app/core.py` concentra configuracoes, CORS, rate limit, autenticacao opcional por `X-API-Key`, modelos Pydantic e fila de reprocessamento.
- Chat: `routers/rag.py` expoe `/chat`, valida modelo Ollama, recupera historico em memoria e chama `answer_question_stream`.
- Busca semantica: `routers/rag.py` expoe `/search`, que chama `retriever.search`.
- Pipeline de resposta: `bbsia/rag/pipeline.py` executa busca, monta contexto, gera prompt, chama Ollama e aplica fallback extrativo em alguns cenarios.
- Geracao: `bbsia/rag/generation/generator.py` valida endpoint/modelo Ollama e envia prompt via `/api/generate`.
- Recuperacao: `bbsia/rag/retrieval/retriever.py` carrega metadados, modelo de embedding e Qdrant local, aplica filtros, busca dense no Qdrant, busca sparse BM25 local, funde via RRF, reranqueia e deduplica por `parent_id`.
- Vetor store: `bbsia/infrastructure/vector_store.py` encapsula Qdrant local em `data/qdrant_db`, colecao `bbsia_chunks`.
- Ingestao: `/reprocessar` em `routers/admin.py` aciona `extrator.run_extraction`, `chunking.run_chunking`, `embedding.run_embedding` e `reload_resources`.
- Extracao: `bbsia/rag/ingestion/extrator.py` le PDFs da raiz, `docs/` e `uploads/approved`; usa Docling quando disponivel, fallback PyMuPDF/OCR, e registra metadados via `classificador.py`.
- Chunking: `bbsia/rag/ingestion/chunking.py` gera chunks parent-child a partir de `data/documentos_extraidos_v2.json`.
- Embeddings: `bbsia/rag/ingestion/embedding.py` gera embeddings E5, recria Qdrant local e escreve `data/qdrant_index_metadata/metadata.json`.

## Estado atual dos dados

- `data/chunks.json`: 638 chunks.
- `data/qdrant_index_metadata/metadata.json`: 638 chunks, modelo `intfloat/multilingual-e5-large`, dimensao 1024.
- Distribuicao por area no indice atual: `juridico` 386, `ia` 170, `infraestrutura` 55, `tecnologia` 27.
- Distribuicao por tipo de conteudo: `text` 491, `table` 147.
- `data/biblioteca.json`: 8 documentos catalogados.
- `content_type = solucao_piloto`: nenhum chunk encontrado.
- `data/parents.json`: ausente na copia investigada; `metadata.json` tambem registra `parents` vazio. Na pratica, o contexto atual tende a usar apenas `texto` do child chunk, nao o parent completo.
- `data/qdrant_db/meta.json`: colecao `bbsia_chunks` existe com vetores 1024 e distancia Cosine.
- `data/qdrant_index_metadata/manifest.json`: contem caminho absoluto para outra pasta (`.../BBSIA/data/...`) e nao para `.../BBSIA - Copia/...`; risco de auditoria/portabilidade, embora o runtime atual use `metadata.json` diretamente.
- `uploads/metadata_uploads.json`: contem varias entradas em quarentena, mas as pastas `uploads/quarantine` e `uploads/approved` estao vazias na copia; ha metadados pendurados.

## O que ja ajuda no desafio

- RAG semantico local ja existe e esta operacional no desenho: dense retrieval, BM25, RRF e reranker.
- Prompt ja e conservador: exige resposta em portugues, uso do contexto e citacoes.
- Ha fallback extrativo quando o LLM falha ou quando a resposta declara falta de evidencia apesar de haver contexto.
- Ha validacao basica de upload PDF, limites de tamanho/paginas/caracteres e varredura simples de prompt injection em uploads.
- Ha endpoints de administracao, status, filtros, biblioteca, busca e chat streaming.
- Ha cobertura razoavel de testes para API, filtros, pipeline, fallback, calibracao de threshold, embedding/chunking e catalogo de solucoes.

## Lacunas para "solucao de problemas"

- Falta um catalogo real de solucoes: `bbsia/domain/catalogo/service.py` espera `catalogo/solucoes_piloto.json`, que esta sendo criado agora.
- O catalogo de solucoes nao alimenta o indice oficial. O `/chat` e `/search` leem `data/qdrant_index_metadata/metadata.json`, nao `data/solucoes_piloto_chunks.json` nem `data/solucoes_faiss_index`.
- O schema de solucao ainda e generico. Para solucao de problemas, faltam campos como sintomas, contexto de uso, pre-condicoes, causa-raiz, restricoes, custo/complexidade, riscos, passos de implantacao, evidencias, orgao/setor aplicavel, maturidade e criterios de escolha.
- `ChatRequest` aceita pergunta, modelo, top_k, filtros de area/assunto e conversation_id. Nao ha contrato para problema estruturado, severidade, restricoes do usuario, tipo de organizacao ou formato de recomendacao.
- `bbsia/rag/pipeline.py` retorna resposta textual e fontes; nao retorna recomendacoes estruturadas, score de adequacao, alternativas, trade-offs ou perguntas de esclarecimento.
- `retriever.search` ranqueia chunks documentais; nao ha estrategia de ranking voltada a matching problema -> solucao.
- `bbsia/rag/retrieval/query_planning.py` e uma heuristica simples, desligada por padrao (`ENABLE_QUERY_PLANNING=false`) e limitada a keywords fixas.
- A checagem de faithfulness streaming esta desligada por padrao, e a checagem NLI sincrona tambem depende de flag. Para recomendacao de solucoes, isso aumenta risco de recomendacoes nao suportadas.

## Arquivos que provavelmente precisarao ser alterados

- `schemas/solucao_piloto.schema.json`: expandir o modelo de solucao para representar problema, contexto, restricoes, aplicabilidade, evidencias e riscos.
- `bbsia/domain/catalogo/service.py`: transformar o catalogo de solucoes em fonte de dominio de primeira classe, com validacao, normalizacao e chunks otimizados para matching de problemas.
- `catalogo/solucoes_piloto.json`: popular a fonte real de solucoes (arquivo ja criado).
- `bbsia/rag/ingestion/embedding.py`: separar ou parametrizar melhor a indexacao de documentos e solucoes para evitar sobrescrita acidental do Qdrant oficial.
- `scripts/gerar_embeddings_solucoes.py`: corrigir a estrategia de indexacao; hoje o script fala em FAISS e chama `run_embedding`, que ainda recria `data/qdrant_db`.
- `bbsia/rag/retrieval/retriever.py`: adaptar busca/ranking para distinguir documentos, solucoes e possiveis tipos de resposta; incluir filtros e metadados de solucao.
- `bbsia/infrastructure/vector_store.py`: avaliar colecoes separadas no Qdrant, por exemplo `bbsia_chunks` e `bbsia_solucoes`, ou payloads bem tipados em uma unica colecao.
- `bbsia/rag/pipeline.py`: criar uma orquestracao voltada a solucao de problemas, possivelmente com etapas de entendimento do problema, recuperacao de solucoes, recuperacao de evidencias e resposta estruturada.
- `bbsia/rag/generation/generator.py`: ajustar prompt para recomendacao baseada em evidencias, com limites claros, alternativas e perguntas de esclarecimento quando a entrada for insuficiente.
- `bbsia/rag/generation/faithfulness.py`: fortalecer avaliacao de grounding para recomendacoes e nao apenas respostas factuais curtas.
- `bbsia/rag/retrieval/query_planning.py`: evoluir para self-query/roteamento por intencao, problema, area, maturidade e restricoes.
- `routers/rag.py`: expor contrato especifico para recomendacao de solucoes ou enriquecer `/chat` com tipos de evento/metadata de recomendacao.
- `bbsia/app/core.py`: adicionar modelos Pydantic para problema/solucao, se a API passar a aceitar payload estruturado.
- `tests/test_catalogo_solucoes.py`: expandir testes do schema de solucao.
- `tests/test_rag_engine.py`, `tests/test_pipeline_integration.py`, `tests/test_filters.py`: cobrir ranking e geracao orientados a solucao.
- `benchmarks/eval_dataset.json` e `benchmarks/rag_benchmark.py`: incluir queries de problema com solucao esperada, nao apenas recuperacao documental.
- `.env.example` e `config.py`: revisar flags de query planning, reranker, faithfulness e colecoes/indices.
- `README.md`: documentacao operacional agora existe, porem precisa ser mantida atualizada com a nova arquitetura e o RAG semantico.

## Riscos arquiteturais

- Risco de sobrescrita de indice: `bbsia/rag/ingestion/embedding.py` sempre usa `data/qdrant_db`; `scripts/gerar_embeddings_solucoes.py` passa outro `index_dir`, mas isso so altera o diretorio de metadata, nao o Qdrant local. Rodar esse script pode substituir o indice principal por chunks de solucoes.
- Risco de dominio incompleto: o RAG atual recupera documentos, nao solucoes acionaveis. Sem schema de problema e criterios de matching, o LLM pode transformar trechos documentais em recomendacoes fracas.
- Risco de contexto curto: `parents` esta vazio no indice atual, entao o sistema pode perder contexto de parent chunk e responder com trechos muito locais.
- Risco de inconsistencia historica FAISS/Qdrant: ainda ha nomes e caminhos legados (`faiss_index`, `solucoes_faiss_index`, `faiss_vectors`) apesar de Qdrant ser o backend oficial.
- Risco de modelos em ambiente restrito: `HF_LOCAL_FILES_ONLY=true` vale para embeddings no retriever, mas o reranker usa `local_files_only=False`; NLI tambem nao segue claramente a mesma politica. Isso pode quebrar ambiente offline ou tentar rede sem querer.
- Risco de seguranca de prompt injection: uploads passam por varredura simples, mas documentos em `docs/` entram no corpus sem a mesma etapa de quarentena. O conteudo recuperado entra cru no prompt.
- Risco de historico em memoria: `conversation_id` usa dicionario global sem TTL/persistencia/limite por conversa alem do recorte no prompt; em multi-worker ou uso prolongado, ha risco de inconsistencia e crescimento.
- Risco de autenticacao em producao: `.env.example` deixa chaves vazias; nesse modo, endpoints sensiveis dependem apenas de rede/ambiente.
- Risco de metadados pendurados: `uploads/metadata_uploads.json` referencia arquivos de quarentena que nao existem mais nesta copia.
- Risco de manifesto nao portavel: `manifest.json` aponta para caminho absoluto fora desta copia, o que pode confundir auditoria, diagnostico e automacoes.

## Diagnostico final

- A fundacao de RAG semantico esta bem encaminhada para consulta documental.
- A arquitetura Clean Architecture/modularizacao foi concluida, isolando os dominios na pasta `bbsia/` (app, core, domain, infrastructure, rag). As antigas fachadas na raiz do projeto (`rag_engine.py`, `api.py`) foram removidas ou movidas, reduzindo a divida tecnica de acoplamento.
- A proxima evolucao arquitetural deve focar 100% no dominio de problemas e solucoes: o ponto critico nao e adicionar mais LLM, e sim criar um contrato de dominio para "problema -> evidencias -> solucoes candidatas -> recomendacao".
- Antes de implementar features novas nesse novo modelo, vale corrigir a estrategia de indexacao de solucoes para nao competir com o indice documental oficial do Qdrant.
