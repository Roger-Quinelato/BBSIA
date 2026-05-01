# SPEC.md: Chatbot com RAG Semântico para Solução de Problemas

## 1. O Problema a ser Resolvido
Atualmente, o sistema BBSIA possui um mecanismo de RAG (Retrieval-Augmented Generation) funcional e maduro para responder a perguntas baseadas em documentos indexados. No entanto, a arquitetura atual tem uma deficiência crítica: ela resolve o problema de "perguntar aos documentos", mas não resolve adequadamente o cenário de "diagnosticar um problema e recomendar uma solução estruturada". Existe um catálogo embrionário de "soluções piloto", mas ele não está integrado ao índice principal nem ao fluxo de resposta, fazendo com que o chatbot careça de um modelo de domínio focado em problemas e não consiga emitir recomendações baseadas em um catálogo validado de soluções.

## 2. Objetivos de Negócio
- **Evolução do Caso de Uso:** Evoluir o chatbot de um mero "assistente de leitura de documentos" para uma ferramenta de diagnóstico e recomendação de soluções para problemas complexos.
- **Confiabilidade nas Recomendações:** Garantir que as soluções recomendadas sejam estritamente baseadas em um catálogo curado ("soluções piloto"), evitando alucinações e recomendações inseguras.
- **Padronização das Soluções:** Estabelecer um contrato claro e detalhado para o que constitui uma solução (sintomas, causa-raiz, riscos, restrições, etc.), permitindo escalabilidade na adição de novo conhecimento.

## 3. User Stories
- **US01:** Como usuário enfrentando um problema técnico/operacional, quero descrever os sintomas no chat para que a IA diagnostique a possível causa raiz baseada no catálogo de soluções.
- **US02:** Como usuário, quero receber recomendações de soluções estruturadas (incluindo pré-condições, passos e riscos) para que eu possa aplicar a resolução com segurança.
- **US03:** Como administrador do conhecimento, quero gerenciar um catálogo JSON de "soluções piloto" independente dos documentos gerais, para que o chatbot consulte fontes especializadas ao recomendar ações.
- **US04:** Como engenheiro do sistema, quero que as buscas por soluções não poluam ou sobrescrevam o índice de documentos gerais, garantindo a estabilidade de ambos os fluxos de consulta.

## 4. Fora de Escopo (Out of Scope)
- **Substituição do Motor de LLM:** A inferência continuará sendo feita pelo Ollama local. Não migraremos para provedores de nuvem neste escopo.
- **Reescrita Total (Rewrite):** A infraestrutura existente (FastAPI, Qdrant local, SentenceTransformers) será reutilizada. Refatorações extremas que quebrem os endpoints atuais `/chat` e `/search` estão proibidas.
- **Geração Autônoma de Soluções:** O LLM *não* deve inventar passos de solução que não estejam ancorados no catálogo de soluções piloto ou nos documentos de referência.
- **Gerenciamento de Estado Complexo de Longo Prazo:** Além do contexto imediato da conversa na memória (já implementado), não implementaremos persistência de histórico de sessões em banco de dados neste ciclo.
- **Desenvolvimento de Interface Gráfica (Frontend):** O foco exclusivo desta etapa é a inteligência do RAG e os contratos da API backend.
