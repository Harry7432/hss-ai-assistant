# Auditoria Técnica do Código Existente

**Projeto**: `hss-ai-assistant`  
**Data da Auditoria**: 30/09/2026  
**Status**: Concluída, Revisada e Calibrada (v2.1)  
**Spec Reference**: `specs/001-modular-rag-engine/spec.md`  
**Plan Reference**: `specs/001-modular-rag-engine/plan.md`  
**Tasks Reference**: `specs/001-modular-rag-engine/tasks.md`  

---

## 1. Resumo Executivo

O projeto atual consiste em um protótipo inicial de RAG (Retrieval-Augmented Generation) estruturado em dois scripts planos (`criar_db.py` e `main.py`), uma pasta de documentos (`base/`) e um arquivo de configuração de variáveis (`env`).

A arquitetura modular desacopla responsabilidades no pacote `src/`. Esta auditoria técnica identifica os gargalos do código legado e consolida as decisões arquiteturais mandatórias para a implementação:
1. **Indexação não idempotente e risco de corrupção**: Ausência de manifesto com `paths: list` por `doc_hash`, falta de identificadores determinísticos `{doc_hash}_{chunk_index}`, ausência de reconciliação inicial manifesto x ChromaDB (onde divergência deve forçar reindexação total), gravação não atômica do manifesto (que deve ser gravado por último via `tmp` + rename), tratamento falho de renomeação vs cópia (renomeação atualiza metadados sem re-embedar; cópia emite warning e é ignorada), falta de controle de fingerprint da configuração (incluindo `distance_metric`, com nome de coleção derivado do fingerprint e não duplicado nele, nova coleção passando a valer na gravação atômica do manifesto e remoção da coleção antiga protegida por `--prune`) e ordem arriscada de inserção/deleção.
2. **Interpretação falha de métricas vetoriais**: O código trata scores de similaridade como probabilidades bayesianas de acerto. A coleção deve ser formalmente configurada com distância cosseno via `configuration={"hnsw": {"space": "cosine"}}` (confirmado para ChromaDB 1.5.9), onde `score = distância` (menor é melhor), e o `RELEVANCE_THRESHOLD` atua como distância máxima permitida calibrado experimentalmente via Task 3.3 (utilizando conjunto pequeno e separado de `eval_dataset.json`, rodando fora do pytest padrão via script ou marker por usar a API real de embeddings) antes da Fase 5. A ausência de contexto relevante deve ser retorno estruturado, não exceção.
3. **Vulnerabilidades de prompt injection e alucinação de fontes**: Ausência de delimitação `<context>`, interpolação vulnerável (o escape de tags `</context>` e `<context>` deve ocorrer estritamente na montagem do prompt, preservando o `content` do chunk original intacto), falta de validação de segurança via documento-fixture envenenado com palavra-canário (indexado exclusivamente em diretório e coleção temporários, nunca em `db/`), falta de sentinela no prompt para recusa, recontextualizador desprotegido, snippets de citação que devem ser literais do conteúdo original e risco de vazamento de credenciais `sk-proj-...` ou texto de documentos em logs.
4. **Isolamento de custos, resiliência e avaliação de qualidade**: Necessidade de retries nativos (`max_retries=3`) no cliente OpenAI aceitando retries em HTTP 429 de cota esgotada e mapeando a falha final para `QuotaExceededError` testada com transporte HTTP simulado (com simulação de sleep ou `max_retries=0` no cliente de teste para não estourar o limite de 10s do SC-001); testes unitários 100% offline com `DeterministicFakeEmbedding` e coleções com nomes únicos (UUID); criação de tarefas de teste dedicadas para telemetria (`test_telemetry.py`), `chat_loop` (`test_chat_loop.py`), wrappers CLI (`test_cli_wrappers.py`) e benchmark determinístico local SC-014 (`test_performance.py`); e suíte RAG Eval isolada com gate por contagem bruta sem limiares absolutos prévios, baseline após primeira execução real sob temperatura 0 com regra única de regressão (falha se qualquer métrica perder mais de N amostras em relação ao baseline, com N=1 inicial), validade de citação na saída bruta do LLM, gabarito por trecho-âncora e fidelidade postergada para a v2.

---

## 2. Análise Crítica por Componente

### 2.1. Ingestão e Indexação (`criar_db.py`)

| Aspecto | Situação Atual | Impacto / Risco | Diagnóstico e Correção Necessária |
| :--- | :--- | :--- | :--- |
| **Execução em Nível de Módulo e Contrato CLI** | `criar_db()` é chamado diretamente na linha 36. | Re-executa todo o pipeline ao ser importado por outros módulos ou testes. | Encapsular execução em bloco `if __name__ == "__main__":` e definir contrato CLI aceitando exclusivamente `--base-dir` e `--prune` (removidos `--chunk-size/--chunk-overlap` para manter governança estrita via `Settings` e fingerprint), retornando código de saída 0 em sucesso e != 0 em erro. (FR-015, SC-008) |
| **Manifesto, Fingerprint e Reconciliação** | Inexistente. Apenas adição cega no ChromaDB. | Mudanças em configurações misturam dados incompatíveis; corrupção se o processo cair no meio. | Criar manifesto (`manifest.json`) armazenando `paths: list` por `doc_hash` e fingerprint global `{"embedding_model", "chunk_size", "chunk_overlap", "splitter_version", "distance_metric"}` (com nome de coleção derivado do fingerprint e não duplicado nele). Se o fingerprint mudar, provisionar coleção nova no ChromaDB com nome derivado e reindexar acervo. A nova coleção passa a valer na gravação atômica do manifesto por último (`tmp` + rename), e a coleção antiga obsoleta é removida com proteção da flag `--prune`. No início de cada sincronização, reconciliar manifesto x Chroma (divergência = reindexação total). (FR-005, SC-002, SC-003) |
| **Identificadores Determinísticos (IDs)** | Sem IDs explícitos (gerados aleatoriamente pelo Chroma). | Impossibilita sincronização e rastreabilidade determinística. | Utilizar hash SHA-256 completo do conteúdo do arquivo (`doc_hash`) + `chunk_index` (`chunk_id = f"{doc_hash}_{chunk_index}"`). O cálculo redundante de hash por chunk individual deve ser removido. (FR-004, SC-002) |
| **Renomeação vs Cópia de Arquivos** | Nomes de arquivos idênticos duplicam vetores ou sobrescrevem incorretamente. | Custos repetidos de embeddings e inconsistência em metadados. | Distinguir formalmente: Rename = caminho antigo ausente no disco + novo presente com mesmo `doc_hash` -> atualiza `paths`, `file_name` e `source` sem recalcular embeddings (0 chamadas API). Cópia = ambos caminhos presentes no disco com mesmo `doc_hash` -> registrar log de warning e ignorar a cópia sem duplicar chunks. (FR-005, SC-003) |
| **Atualização Segura e Trava de Poda** | Inexistente. Deleção de órfãos sem validação. | Risco de perda de dados se o processo falhar no meio, ou exclusão acidental em massa por falha de montagem ou disco. | Na atualização, inserir chunks novos ANTES de apagar antigos. Poda de órfãos restrita a arquivos confirmadamente ausentes no disco (nunca para falhas de leitura). Trava de segurança: recusar apagar mais de 20% (`MAX_PRUNE_PERCENTAGE`) do índice sem a flag explícita `--prune`. (FR-005, SC-003) |
| **Parâmetros de Chunking** | Valores fixos no código (`chunk_size=2000`, `chunk_overlap=500`). | Dilui a densidade semântica para embeddings modernos. | Padronizar valores padrão configuráveis via `.env`: `CHUNK_SIZE=1000` e `CHUNK_OVERLAP=200`. (FR-001, FR-004) |
| **Metadados, Numeração de Páginas e Robustez** | Metadados básicos sem convenção uniforme; quebra em PDFs sem texto. | Páginas 0-indexed confundem o usuário; falha em PDFs escaneados ou diretório vazio. | Adotar rigorosamente numeração 1-indexed para páginas (`page: int`). Tratar PDFs escaneados/sem texto emitindo log de warning e pulando o arquivo sem abortar. Tratar diretório `base/` vazio emitindo warning e finalizando com 0 documentos sem erro. (FR-003, SC-003) |

---

### 2.2. Recuperação Vetorial (`main.py` - Linhas 24-37)

| Aspecto | Situação Atual | Impacto / Risco | Diagnóstico e Correção Necessária |
| :--- | :--- | :--- | :--- |
| **Acoplamento da Busca** | Recuperação embutida em `perguntar()`, misturada com UI e LLM. | Impossibilita testar a qualidade da busca semântica isoladamente. | Extrair para `src/retrieval/search.py` retornando lista tipada de `RetrievedChunk`. (FR-006) |
| **Métrica e Interpretação de Scores** | `if len(resultados) == 0 or resultados[0][1] < 0.7:` assumindo similaridade direta. | Trata score arbitrariamente como probabilidade de correção. Coleção padrão pode usar L2 ou cosseno sem clareza. | Configurar explicitamente a coleção Chroma com distância cosseno via `configuration={"hnsw": {"space": "cosine"}}` (confirmado para ChromaDB 1.5.9). O score DEVE ser a distância cosseno (MENOR é melhor, 0 = idêntico). Tratar score estritamente como heurística geométrica. `RELEVANCE_THRESHOLD` é calibrado previamente via Task 3.3 em rotina retrieval-only com dataset pequeno e separado do de avaliação, fora do pytest padrão com API real antes da Fase 5. (FR-006, FR-007, SC-004) |
| **Tratamento de Ausência de Resultados** | Mensagem estática em `print()`. | Falta de retorno estruturado para orquestração. | Ausência de contexto relevante (score acima do threshold ou banco vazio) DEVE ser retorno estruturado normal (lista vazia e `is_fallback=True`) e NUNCA lançamento de exceção. (FR-002, FR-006) |
| **Preservação de Metadados** | Apenas `page_content` é extraído para montar `base_conhecimento`. `metadata` é descartado. | Arquivo e página são perdidos antes da etapa de geração. | Preservar metadados integrais (`file_name`, `page` 1-indexed, `chunk_id`, `doc_hash`) em `RetrievedChunk`. (FR-006, SC-004) |

---

### 2.3. Geração, Segurança e Guardrails (`main.py` - Linhas 12-46)

| Aspecto | Situação Atual | Impacto / Risco | Diagnóstico e Correção Necessária |
| :--- | :--- | :--- | :--- |
| **Isolamento de Contexto, Escape e Injeção com Canário** | Interpolação direta sem tags delimitadoras. | Vulnerabilidade a **Prompt Injection Indireto** via instruções adversárias embutidas nos PDFs. | Delimitar contexto estritamente em `<context>...</context>`. O escape de tags `</context>` e `<context>` DEVE ocorrer estritamente durante a montagem do template de prompt, preservando integralmente o atributo `content` original do chunk. Mitigação comprovada via teste com documento-fixture envenenado contendo palavra-canário oculta (indexado exclusivamente em diretório e coleção temporários, nunca em `db/`). Tratar também input e histórico como não confiáveis no recontextualizador. (FR-008, FR-013, SC-009, SC-012) |
| **Garantia de Recusa e Frase-Sentinela** | Instrução frágil no prompt sem sentinela fixa. | Modelo tenta inferir respostas não fundamentadas (alucinação). | Instruir o modelo a responder com frase-sentinela fixa de recusa quando o contexto não contiver informações para responder à pergunta. A detecção da sentinela ativa `is_fallback=True`. (FR-008, FR-010, SC-006) |
| **Validação Estrita de Citações e Snippets** | Nenhuma citação é gerada nem validada. | Usuário não consegue auditar fontes; modelo pode citar referências fantasmas. | Resposta não-fallback exige pelo menos uma citação válida `[N]`. Marcadores inválidos são removidos do texto; se sobrar zero citações válidas, converter para fallback (`is_fallback=True`). "Fontes Consultadas" lista EXCLUSIVAMENTE fontes citadas no texto. Snippet deve ser trecho literal truncado extraído diretamente do `content` original do chunk. Páginas 1-indexed. (FR-009, SC-005) |
| **Resiliência e Retries da API OpenAI** | Chamada direta sem retries nem tratamento de cota. | Instabilidades transitórias derrubam o fluxo; erros permanentes de cota geram loops desnecessários. | Configurar retries nativos (`max_retries=3`) no cliente `ChatOpenAI` e `OpenAIEmbeddings` para erros transitórios (aceitando retries no 429 mesmo em cenário de cota). Mapear a falha final decorrente de cota esgotada (`insufficient_quota`) para `QuotaExceededError`, com validação via transporte HTTP simulado (com simulação de sleep ou `max_retries=0` no cliente de teste para não estourar o limite de 10s do SC-001). HTTP 401 falha imediatamente sem retry. (FR-011, SC-010) |
| **Privacidade em Logs e Telemetria** | Prints abertos no terminal sem sanitização. | Risco de vazamento de credenciais e exposição indevida de dados dos documentos. | Logger JSON estruturado omitindo conteúdo textual de documentos por padrão (somente IDs, tamanhos e latências). Sanitizador regex cobrindo formatos `sk-proj-...` e `sk-...`. Telemetria LangSmith estritamente opt-in (`LANGCHAIN_TRACING_V2=false`), com testes unitários dedicados em `tests/unit/test_telemetry.py`. (FR-014, NFR-003, SC-007, SC-013) |

---

### 2.4. Estratégia de Testes, TDD e Observabilidade

| Aspecto | Situação Atual | Impacto / Risco | Diagnóstico e Correção Necessária |
| :--- | :--- | :--- | :--- |
| **Testes Unitários Offline (TDD)** | Zero testes existentes. | Violação de TDD e risco contínuo de regressão. | Implementar testes unitários 100% offline com `DeterministicFakeEmbedding`, `FakeListLLM` e coleções Chroma com nomes únicos (UUID), 0 custo de API e execução em menos de 10s via `pytest -m "not eval"`. (NFR-001, SC-001) |
| **Desempenho Local Determinístico (NFR-004)** | Especificação anterior incluía chamadas de LLM em medição de tempo local ("hardware padrão"). | Não mensurável de forma determinística devido à latência variável de rede e LLM. | Benchmark automatizado dedicado em `tests/unit/test_performance.py` cobrindo busca vetorial local e montagem de prompt com validação de citações (< 200ms em hardware mensurável de referência). (NFR-004, SC-014) |
| **Avaliação de Qualidade do RAG (RAG Eval)** | Inexistente. | Falta de método para validar precisão, recusa e citações. | Criar suíte RAG Eval isolada com marker `@pytest.mark.eval` em `tests/eval/test_rag_quality.py` com dataset de 25+ amostras (`eval_dataset.json` com gabarito por trecho-âncora e documento envenenado com palavra-canário indexado em diretório e coleção temporários, nunca em `db/`). Gate inicial por contagem bruta sem limiares absolutos prévios. Baseline (`eval_baseline.json`) gravado sob temperatura 0 após a primeira execução real com regra única de regressão: falha se qualquer métrica perder mais de N amostras em relação ao baseline (N=1 inicial). Validade de citação aferida sobre a saída bruta do LLM. Avaliação de Fidelidade (Faithfulness / LLM-as-a-judge) postergada para a v2. (SC-016) |
| **Contrato e Testes de Adaptadores Legados** | Scripts planos sem interface de terminal estruturada nem testes. | Dificuldade de acoplamento com CLI ou interfaces futuras. | Definir contrato de CLI para `criar_db.py` (--base-dir, --prune) e `main.py` (--session-id) e implementar suítes de testes unitários dedicadas em `tests/unit/test_cli_wrappers.py` e `tests/unit/test_chat_loop.py`. (FR-015, SC-008) |

---

## 3. Matriz de Riscos Atualizada

| Risco | Probabilidade | Impacto | Estratégia de Mitigação |
| :--- | :--- | :--- | :--- |
| **Prompt Injection Indireto e Delimitadores** | Média | Alto | Delimitação do contexto em `<context>`, escape de `</context>` e `<context>` estritamente na montagem do prompt mantendo `content` original, System Prompt imutável, isolamento no recontextualizador e validação com documento envenenado contendo palavra-canário oculta (indexado exclusivamente em diretório e coleção temporários, nunca em `db/`). (FR-008, SC-009) |
| **Alucinação e Citações Fantasma** | Média | Alto | Exigência de >= 1 citação válida para respostas não-fallback, expurgo de citações inválidas com fallback automático para zero citações, fontes listando apenas citados, snippets literais do texto original e medição de validade de citação na saída bruta do LLM. (FR-009, SC-005, SC-016) |
| **Custos Inesperados em Testes e Avaliações** | Alta | Médio | Mocks obrigatórios (`DeterministicFakeEmbedding`, `FakeListLLM`) em 100% dos testes unitários; suíte de RAG Eval isolada pelo marker `@pytest.mark.eval` com exclusão automática no comando padrão do pytest (`addopts = "-m 'not eval'"`). (NFR-001, SC-001) |
| **Corrupção de Índice, Divergência e Expurgo** | Alta | Alto | Manifesto com `paths: list` por `doc_hash` gravado por último (`tmp` + rename), reconciliação inicial manifesto x Chroma (divergência = reindex total), detecção de rename (antigo ausente, novo presente) vs cópia (warning e ignorar), fingerprint sem `collection_name` duplicado gerando coleção nova que passa a valer na gravação do manifesto, e remoção da coleção antiga protegida por `--prune`. (FR-005, SC-002, SC-003) |
| **Interpretação Equivocada de Scores** | Alta | Médio | Configuração de distância cosseno via `configuration={"hnsw": {"space": "cosine"}}` (confirmado para ChromaDB 1.5.9), score como distância (menor é melhor), calibração retrieval-only do threshold na Task 3.3 com dataset pequeno e separado fora do pytest padrão antes da Fase 5 e documentação formal do score como heurística geométrica. (FR-006, FR-007, SC-004) |
| **Erros de Cota e Instabilidade da API** | Média | Alto | Retries nativos no cliente OpenAI (`max_retries=3`) aceitando retries em HTTP 429 mesmo em cenário de cota esgotada; mapeamento final de `insufficient_quota` para `QuotaExceededError` testado com transporte HTTP simulado (com simulação de sleep ou `max_retries=0` no cliente de teste para não estourar o limite de 10s do SC-001); e falha imediata em HTTP 401. (FR-011, SC-010) |
| **Vazamento de Credenciais e Conteúdo em Logs** | Média | Crítico | Sanitização de chaves `sk-proj-...` e `sk-...` via regex, exclusão do texto de documentos nos logs por padrão, LangSmith configurado como opt-in com testes dedicados em `tests/unit/test_telemetry.py`. (FR-014, NFR-003, SC-007, SC-013) |

---

## 4. Fronteiras de Escopo e Evolução Futura

Para manter foco e estabilidade na feature `001-modular-rag-engine`, os seguintes recursos são formalmente documentados como fora de escopo imediato e constituem evolução arquitetural futura:
1. **Framework Web / API**: FastAPI, endpoints REST e Swagger interativo.
2. **Banco Corporativo**: Migração para PostgreSQL com `pgvector`.
3. **Mensageria e Filas**: RabbitMQ, Celery, Redis para ingestão assíncrona.
4. **Conteinerização**: Dockerfile e `docker-compose.yml`.
5. **Agentes com LangGraph**: Tool calling e grafos conversacionais complexos.
6. **Avaliação Automatizada de Fidelidade (Faithfulness / LLM-as-a-judge)**: Avaliação de alucinação e conformidade factual por meio de segundo LLM juiz (postergada para a v2).

A arquitetura modular construída nesta feature fornece interfaces desacopladas e tipadas, permitindo acoplar esses adaptadores futuros sem necessidade de refatorar o núcleo do motor RAG.
