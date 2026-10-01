# Tasks: Arquitetura RAG Modular com TDD e Observabilidade

**Feature Branch**: `001-modular-rag-engine`  
**Versão**: 2.1 (Planejamento e Especificação Calibrados - Tarefas Pendentes de Verificação)  
**Spec**: `specs/001-modular-rag-engine/spec.md`  
**Plan**: `specs/001-modular-rag-engine/plan.md`  
**Audit**: `specs/001-modular-rag-engine/audit.md`  

---

### Matriz de Rastreabilidade Completa (Requisitos x Fases x Tarefas x Critérios de Sucesso)

| Requisito / NFR | User Story | Fase | Tarefas Correspondentes | Status | Critério de Sucesso (SC) | Critério de Verificação |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FR-001** | Fundação | Fase 1 | 1.1, 1.4, 1.5 | [x] Concluído | SC-001 | `test_config.py` valida variáveis, tipos e padrões (1000/200, threshold calibrado na Task 3.3, 20% prune) |
| **FR-002** | Fundação | Fase 1, 5 | 1.4, 1.5, 5.1, 5.3 | [ ] Pendente | SC-001, SC-010 | `test_config.py` e `exceptions.py` tratam exceções; ausência de contexto é retorno estruturado; mapeamento para `QuotaExceededError` |
| **FR-003** | US 1 (P1) | Fase 2 | 2.1, 2.2 | [x] Concluído | SC-003 | `test_loader.py` valida páginas 1-indexed, skip gracioso em PDF sem texto e pasta `base/` vazia |
| **FR-004** | US 1 (P1) | Fase 2 | 2.3, 2.4 | [x] Concluído | SC-002 | `test_splitter.py` valida chunks (1000/200), `doc_hash` SHA-256 completo e IDs `{doc_hash}_{chunk_index}` |
| **FR-005** | US 1 (P1) | Fase 2 | 2.5, 2.6 | [ ] Pendente | SC-002, SC-003 | `test_vector_store.py` e `test_persistence.py` validam `paths: list`, rename vs cópia, tmp+rename por último, reconciliação, fingerprint sem duplicar `collection_name`, coleção nova ativa no manifesto e remoção antiga via `--prune` |
| **FR-006** | US 2 (P2) | Fase 3 | 3.1, 3.2, 3.3 | [ ] Pendente | SC-004 | `test_retrieval.py` valida cosseno via `configuration={"hnsw": {"space": "cosine"}}` (1.5.9), ordenação e calibração na Task 3.3 (dataset separado fora do pytest padrão com API real) |
| **FR-007** | US 2 (P2) | Fase 3 | 3.1, 3.2 | [ ] Pendente | SC-004 | Asserções e documentação tratam score estritamente como heurística geométrica de proximidade |
| **FR-008** | US 3, US 4 | Fase 4, 7 | 4.3, 4.4, 7.3, 7.4 | [ ] Pendente | SC-009 | `test_prompt_templates.py` valida escape só no prompt mantendo `content` original; eval testa fixture envenenada com palavra-canário (isolada fora de `db/`) |
| **FR-009** | US 3 (P3) | Fase 4 | 4.1, 4.2 | [ ] Pendente | SC-005 | `test_citations.py` valida >=1 citação em não-fallback, expurgo de inválidas, fontes citadas e snippets literais do texto original |
| **FR-010** | US 4 (P4) | Fase 5 | 5.1, 5.3 | [ ] Pendente | SC-006 | `test_guardrails.py` valida `is_fallback=True` para threshold calibrado, sentinela do modelo e zero citações |
| **FR-011** | US 4 (P4) | Fase 5 | 5.1, 5.3 | [ ] Pendente | SC-010 | `test_guardrails.py` valida retries nativos (`max_retries=3`) em 429 de cota e teste com transporte HTTP simulado (sleep mockado ou max_retries=0 para SC-001 < 10s) para `QuotaExceededError` |
| **FR-012** | US 5 (P5) | Fase 6 | 6.1, 6.2 | [ ] Pendente | SC-011 | `test_memory.py` valida segregação absoluta por `session_id` e limite de janela deslizante |
| **FR-013** | US 5 (P5) | Fase 6 | 6.3, 6.4 | [ ] Pendente | SC-012 | `test_recontextualizer.py` valida query autocontida tratando input e histórico como não confiáveis |
| **FR-014** | US 6 (P6) | Fase 5, 7 | 5.1, 5.2, 7.1, 7.2 | [ ] Pendente | SC-007, SC-013 | `test_guardrails.py` valida sanitização `sk-proj-...`; `test_telemetry.py` (7.1) e `telemetry.py` (7.2) validam LangSmith opt-in e privacidade de documentos |
| **FR-015** | Compat. | Fase 8 | 8.1, 8.2, 8.3, 8.4, 8.5 | [ ] Pendente | SC-008 | Contrato CLI de `criar_db.py` (--base-dir, --prune) e `main.py` (--session-id) e testes TDD em `test_chat_loop.py` (8.1) e `test_cli_wrappers.py` (8.3) |
| **NFR-001** | Não Funcional | Todas | 1.4, 2.1, 2.3, 2.5, 3.1, 4.1, 4.3, 5.1, 6.1, 6.3, 7.1, 7.5, 8.1, 8.3, 8.6 | [ ] Pendente | SC-001 | Testes unitários executam 100% offline com `DeterministicFakeEmbedding`, `FakeListLLM` e coleções com nomes únicos (UUID) |
| **NFR-002** | Não Funcional | Fase 2 | 2.5, 2.6 | [ ] Pendente | SC-002 | Reindexação sem alterações gera 0 chamadas de embedding e 0 alterações no ChromaDB; teste lê configuração real da coleção |
| **NFR-003** | Não Funcional | Fase 5 | 5.1, 5.2 | [ ] Pendente | SC-007 | Sanitização regex de credenciais `sk-proj-...` e logs sem conteúdo textual de documentos por padrão |
| **NFR-004** | Não Funcional | Fase 7 | 7.5 | [ ] Pendente | SC-014 | `test_performance.py` valida benchmark determinístico local (busca + montagem de prompt) em < 200ms em CPU 4 cores / 8GB RAM |
| **NFR-005** | Não Funcional | Fase 8 | 8.6 | [ ] Pendente | SC-015 | Cobertura de testes unitários superior a 85% sobre `src/` via `pytest --cov=src` |
| **RAG Eval** | Qualidade | Fase 7 | 7.3, 7.4 | [ ] Pendente | SC-016 | `pytest -m eval` sobre `eval_dataset.json` (trecho-âncora, canário em tmp_path nunca em `db/`): gate por contagem bruta, baseline sob temp 0, regra única de regressão (> N amostras, N=1 inicial), validade na saída bruta, fidelidade v2 |

---

## Checklist de Tarefas Incrementais

> **Aviso de Integridade**: Todas as tarefas iniciam desmarcadas (`[ ]`). Uma tarefa só pode ser marcada como concluída (`[x]`) com evidência colada comprovando aprovação (saída de execução do `pytest`, caminho do arquivo de teste e número de linha).

### Fase 1: Ambiente, Dependências e Configurações Base
- [x] **Task 1.1**: Criar `.env.example` documentando todas as variáveis, padrões calibrados e tipos (`OPENAI_API_KEY`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `RETRIEVAL_K=4`, `RELEVANCE_THRESHOLD=0.35`, `MAX_PRUNE_PERCENTAGE=20`, `LLM_MODEL=gpt-4o-mini`, `EMBEDDING_MODEL=text-embedding-3-small`, `LANGCHAIN_TRACING_V2=false`).
- [x] **Task 1.2**: Criar `requirements.txt` e `pyproject.toml` contendo dependências de produção e testes, registrando formalmente o marker `eval` e definindo `addopts = "-m 'not eval'"` para isolar suíte de avaliação do `pytest` padrão.
- [x] **Task 1.3**: Configurar ambiente virtual local `.venv` e validar execução isolada do interpretador Python.
- [x] **Task 1.4**: Escrever testes unitários em `tests/unit/test_config.py` validando leitura de variáveis, tipos e valores padrão padronizados, e configurar fixtures em `tests/conftest.py` com `DeterministicFakeEmbedding` e coleções Chroma ephemeral com nomes únicos via UUID.
- [x] **Task 1.5**: Implementar `src/core/config.py` e hierarquia de exceções em `src/core/exceptions.py` (`IngestionError`, `VectorStoreUnavailableError`, `APIConnectionError`, `QuotaExceededError`), garantindo que ausência de contexto relevante seja retorno estruturado e não exceção.

> **Evidência de Execução (`tests/unit/test_config.py:7-71`, `tests/unit/test_exceptions.py:11-30`)**:
> ```text
> tests/unit/test_config.py::test_settings_default_values PASSED           [ 12%]
> tests/unit/test_config.py::test_settings_custom_values PASSED            [ 25%]
> tests/unit/test_config.py::test_settings_validation_invalid_chunk_overlap PASSED [ 37%]
> tests/unit/test_config.py::test_settings_validation_invalid_threshold PASSED [ 50%]
> tests/unit/test_config.py::test_settings_validation_invalid_max_prune_percentage PASSED [ 62%]
> tests/unit/test_config.py::test_get_settings_singleton PASSED            [ 75%]
> tests/unit/test_exceptions.py::test_exception_hierarchy PASSED           [ 87%]
> tests/unit/test_exceptions.py::test_raise_custom_exceptions PASSED       [100%]
> ======================== 8 passed, 1 warning in 0.15s =========================
> ```

### Fase 2: Ingestão e Indexação Idempotente com Manifesto, SHA-256 e Sincronização Segura (User Story 1 - P1)
- [x] **Task 2.1**: Escrever testes unitários para o carregador em `tests/unit/test_loader.py` cobrindo extração de páginas 1-indexed, metadados padronizados (`file_name`, `page`, `source`), tratamento gracioso de PDFs sem texto/escaneados (warning e skip) e diretório `base/` vazio (warning e 0 documentos sem erro).
- [x] **Task 2.2**: Implementar `src/indexing/loader.py` garantindo extração resiliente com páginas 1-indexed.
- [x] **Task 2.3**: Escrever testes unitários em `tests/unit/test_splitter.py` validando `CHUNK_SIZE=1000` e `CHUNK_OVERLAP=200`, hash SHA-256 completo do arquivo (`doc_hash`), IDs determinísticos `{doc_hash}_{chunk_index}` e omissão de hash por chunk individual.
- [x] **Task 2.4**: Implementar `src/indexing/splitter.py` integrado a `Settings`.

> **Evidência de Execução (`tests/unit/test_loader.py:9-106`, `tests/unit/test_splitter.py:10-132`)**:
> ```text
> tests/unit/test_loader.py::test_load_directory_nonexistent PASSED        [  9%]
> tests/unit/test_loader.py::test_load_documents_empty_directory PASSED    [ 18%]
> tests/unit/test_loader.py::test_load_documents_success PASSED            [ 27%]
> tests/unit/test_loader.py::test_load_documents_multi_page_1_indexed PASSED [ 36%]
> tests/unit/test_loader.py::test_load_documents_pdf_without_text_ignored PASSED [ 45%]
> tests/unit/test_splitter.py::test_splitter_settings_defaults PASSED      [ 54%]
> tests/unit/test_splitter.py::test_splitter_custom_parameters PASSED      [ 63%]
> tests/unit/test_splitter.py::test_splitter_mandatory_metadata_and_no_chunk_hash PASSED [ 72%]
> tests/unit/test_splitter.py::test_splitter_same_file_generates_same_ids PASSED [ 81%]
> tests/unit/test_splitter.py::test_splitter_altered_file_generates_different_ids PASSED [ 90%]
> tests/unit/test_splitter.py::test_splitter_sequential_chunk_index_across_pages PASSED [100%]
> ======================== 11 passed, 1 warning in 2.35s ========================
> ```

- [x] **Task 2.5 (Manifesto, Resiliência do Loader e Sincronização - Fatia 3)**: Implementação e testes unitários em `tests/unit/test_loader.py` e `tests/unit/test_manifest.py` cobrindo:
  - Carregador com `LoadResult` segregando `documents`, `failed_files` e `skipped_no_text`, sem exceção em arquivo corrompido.
  - Configurações `collection_name` (default "hss_docs") e `distance_metric` ("cosine") em `Settings`, e `SPLITTER_VERSION = "1"`.
  - Estrutura de `IndexManifest` com `fingerprint` (6 campos) e `files: dict` por `doc_hash` com `paths: list`.
  - Gravação atômica (`tmp` + rename) com garantia de recuperação em falha e tratamento resiliente de ausente/corrompido como vazio.
  - Invalidação determinística de fingerprint para qualquer um dos 6 campos alterados.
  - Reconciliação exata com IDs do ChromaDB (divergência em qualquer sentido = reindexação total).
  - Plano de sincronização puro (`SyncPlan`) categorizando `unchanged`, `to_index`, `renamed` (0 re-embeds), `ignored_copies`, `replaced`, `orphans` e `protected` (arquivos com falha de leitura nunca viram órfãos).
  - Trava de poda de segurança (`check_prune_safety`) para `MAX_PRUNE_PERCENTAGE` e documento único.

> **Evidência de Execução (`tests/unit/test_loader.py:9-142`, `tests/unit/test_manifest.py:18-264`)**:
> ```text
> tests/unit/test_loader.py::test_load_directory_nonexistent PASSED        [ 28%]
> tests/unit/test_loader.py::test_load_documents_empty_directory PASSED    [ 30%]
> tests/unit/test_loader.py::test_load_documents_success PASSED            [ 32%]
> tests/unit/test_loader.py::test_load_documents_multi_page_1_indexed PASSED [ 34%]
> tests/unit/test_loader.py::test_load_documents_pdf_without_text_ignored PASSED [ 36%]
> tests/unit/test_loader.py::test_load_documents_corrupted_pdf_recorded_in_failed_files_and_continues PASSED [ 38%]
> tests/unit/test_manifest.py::test_settings_collection_name_and_distance_metric_and_splitter_version PASSED [ 40%]
> tests/unit/test_manifest.py::test_manifest_structure_and_serialization PASSED [ 42%]
> tests/unit/test_manifest.py::test_manifest_atomic_save_and_recovery_on_failure PASSED [ 44%]
> tests/unit/test_manifest.py::test_manifest_load_missing_or_corrupted PASSED [ 46%]
> tests/unit/test_fingerprint_invalidation_each_of_the_6_fields PASSED [ 48%]
> tests/unit/test_reconciliation_with_chroma_equal_missing_and_extra_ids PASSED [ 50%]
> tests/unit/test_sync_plan_unchanged_files PASSED       [ 51%]
> tests/unit/test_sync_plan_to_index_new_and_modified_files PASSED [ 53%]
> tests/unit/test_sync_plan_renamed_file PASSED          [ 55%]
> tests/unit/test_sync_plan_ignored_copies PASSED        [ 57%]
> tests/unit/test_sync_plan_orphans_and_protected_files PASSED [ 59%]
> tests/unit/test_prune_guard_safety_threshold PASSED    [ 61%]
> tests/unit/test_prune_guard_single_document_in_manifest PASSED [ 63%]
> ======================== 52 passed, 1 deselected, 3 warnings in 5.53s ========================
> ```

- [ ] **Task 2.6**: Implementar `src/indexing/vector_store.py` com manifesto (`paths: list`), checagem de fingerprint (sem duplicar nome da coleção), provisionamento de nova coleção derivada, ativação na gravação atômica do manifesto por último (`tmp` + rename), remoção da antiga protegida por `--prune`, reconciliação inicial, inserção prévia, atualização de metadados em renomeados, descarte de cópias com warning e trava de poda.

### Fase 3: Recuperação Vetorial Precisa, Calibrada por Distância Cosseno e Ordenação (User Story 2 - P2)
- [ ] **Task 3.1**: Escrever testes unitários para recuperação vetorial em `tests/unit/test_retrieval.py` cobrindo:
  - Coleção Chroma configurada explicitamente com distância cosseno via `configuration={"hnsw": {"space": "cosine"}}` (confirmado para ChromaDB 1.5.9).
  - Score tratado como distância cosseno (menor score = maior similaridade) e ordenação crescente.
  - Preservação total de metadados com páginas 1-indexed.
  - Filtragem por `RELEVANCE_THRESHOLD` como distância máxima permitida.
  - Retorno estruturado de lista vazia de `RetrievedChunk` sem lançar exceções em banco vazio ou busca sem matches.
  - Asserções impedindo interpretação de scores como probabilidade matemática ou bayesiana.
- [ ] **Task 3.2**: Implementar `src/retrieval/search.py` fornecendo interface de busca por distância cosseno com `configuration={"hnsw": {"space": "cosine"}}` e retorno estruturado.
- [ ] **Task 3.3**: Implementar e executar rotina de calibração do threshold (`RELEVANCE_THRESHOLD`) em modo retrieval-only (sem geração LLM) utilizando conjunto de calibração pequeno e SEPARADO do dataset de avaliação (`eval_dataset.json`), executada fora do pytest padrão (script dedicado `scripts/calibrate_threshold.py` ou marker `@pytest.mark.calibration`) por usar a API real de embeddings para vetorizar as perguntas, ajustando o threshold no `.env` antes da integração dos guardrails na Fase 5.

### Fase 4: Geração de Respostas com Citações e Validação Estruturada (User Story 3 - P3)
- [ ] **Task 4.1**: Escrever testes unitários em `tests/unit/test_citations.py` cobrindo:
  - Exigência de pelo menos uma citação válida `[N]` em respostas que não sejam fallback.
  - Remoção de marcadores inválidos e conversão automática para fallback (`is_fallback=True`) caso restem zero citações válidas.
  - Seção "Fontes Consultadas" contendo apenas as fontes citadas no texto.
  - Snippet literal truncado extraído diretamente do texto original de `RetrievedChunk.content` (nunca gerado pelo LLM) e páginas 1-indexed.
- [ ] **Task 4.2**: Implementar `src/generation/citations.py` com validador e formatador estrito de citações.
- [ ] **Task 4.3**: Escrever testes unitários em `tests/unit/test_prompt_templates.py` validando o isolamento do contexto via tags `<context>...</context>`, escape obrigatório de strings `</context>` e `<context>` estritamente na montagem do prompt (mantendo o atributo `content` do chunk original intacto) e exigência de frase-sentinela fixa para ausência de resposta.
- [ ] **Task 4.4**: Implementar `src/generation/prompt_templates.py` com System Prompt imutável, escape no momento da interpolação e templates com frase-sentinela de recusa.

### Fase 5: Guardrails de Segurança, Mitigação de Injeção e Fallback Estrito (User Story 4 - P4)
- [ ] **Task 5.1**: Escrever testes unitários em `tests/unit/test_guardrails.py` cobrindo:
  - Acionamento imediato de resposta padronizada de recusa com `is_fallback=True` por score acima do threshold calibrado, frase-sentinela do modelo ou ausência de citações válidas.
  - Neutralização de tentativas de injeção de prompt dentro de chunks com tags escapadas na montagem do prompt.
  - Resiliência com retries nativos (`max_retries=3`) no cliente OpenAI para LLM e Embeddings aceitando retries em HTTP 429 mesmo em cenário de cota.
  - Mapeamento da falha final decorrente de cota esgotada (`insufficient_quota`) para `QuotaExceededError`, validado via teste com transporte HTTP simulado (com simulação de sleep ou configuração de `max_retries=0` no cliente de teste para não estourar o limite de 10s do SC-001).
  - Falha imediata sem retry em caso de autenticação inválida (HTTP 401).
  - Sanitização de logs mascarando chaves nos formatos `sk-proj-...` e `sk-...` e omitindo conteúdo textual de documentos por padrão.
- [ ] **Task 5.2**: Implementar `src/observability/logging.py` com formatador JSON estruturado, mascaramento de credenciais e exclusão de texto de documentos por padrão.
- [ ] **Task 5.3**: Implementar `src/generation/rag_engine.py` como orquestrador do fluxo completo com guardrails ativos, retries nativos, política multi-gatilho de fallback e mapeamento de exceções de cota.

### Fase 6: Memória Conversacional Multi-Turn com Recontextualização Segura (User Story 5 - P5)
- [ ] **Task 6.1**: Escrever testes unitários em `tests/unit/test_memory.py` validando isolamento estrito de mensagens por `session_id` e descarte correto de turnos que excedam o limite da janela deslizante (sliding window).
- [ ] **Task 6.2**: Implementar `src/memory/session_store.py`.
- [ ] **Task 6.3**: Escrever testes unitários em `tests/unit/test_recontextualizer.py` validando reescrita de perguntas anafóricas tratando tanto a pergunta do usuário quanto o histórico como entradas não confiáveis com delimitadores de proteção contra prompt injection.
- [ ] **Task 6.4**: Implementar `src/generation/recontextualizer.py` integrando com o `RAGEngine`.

### Fase 7: Observabilidade Segura e Avaliação de Qualidade do RAG (User Story 6 - P6)
- [ ] **Task 7.1**: Escrever testes unitários dedicados em `tests/unit/test_telemetry.py` validando ativação condicional do tracing sob flag opt-in (`LANGCHAIN_TRACING_V2=true`), tratamento de exceções de conexão e garantia de não-envio de texto de documentos.
- [ ] **Task 7.2**: Implementar `src/observability/telemetry.py` para inicializar tracing no LangSmith apenas sob opt-in explícito (`LANGCHAIN_TRACING_V2=true`), documentando envio de dados para servidores externos.
- [ ] **Task 7.3**: Criar `tests/eval/eval_dataset.json` com dataset curado de 25+ amostras com gabaritos baseados em trecho-âncora textual do documento, casos fora de domínio e documento-fixture envenenado contendo palavra-canário oculta (com indexação estrita em diretório e coleção temporários, nunca em `db/`).
- [ ] **Task 7.4**: Implementar suíte de avaliação quantitativa em `tests/eval/test_rag_quality.py` anotada com `@pytest.mark.eval`:
  - Portão inicial de aprovação por contagem bruta sem limiares percentuais absolutos arbitrários antes da consolidação do baseline.
  - Execução sob temperatura 0.
  - Gravação de `tests/eval/eval_baseline.json` exclusivamente após a primeira execução real do pipeline.
  - Regra única de regressão para execuções subsequentes: falha se qualquer métrica perder mais de N amostras em relação ao baseline (N=1 inicial).
  - Medição de Validade de Citação diretamente sobre a saída bruta emitida pelo LLM.
  - Validação de mitigação de prompt injection comprovando que a palavra-canário da fixture envenenada (indexada apenas em diretório e coleção temporários, nunca em `db/`) não é vazada na resposta.
  - Fidelidade documentada formalmente como postergada para a v2.
- [ ] **Task 7.5**: Escrever teste de benchmark automatizado dedicado em `tests/unit/test_performance.py` para validar latência determinística local (< 200ms para busca vetorial + montagem de prompt com validação de citações) conforme SC-014 e NFR-004.

### Fase 8: Adaptadores Legados e Validação Integrada Final
- [ ] **Task 8.1**: Escrever testes unitários dedicados em `tests/unit/test_chat_loop.py` validando parsing de `--session-id`, fluxo interativo de perguntas e respostas e comandos de sessão (`exit`, `clear`, `session`) com mocks de I/O.
- [ ] **Task 8.2**: Implementar CLI interativo em `src/cli/chat_loop.py` com argumentos CLI (`--session-id`) e comandos de sessão (`exit`, `clear`, `session`).
- [ ] **Task 8.3**: Escrever testes unitários dedicados em `tests/unit/test_cli_wrappers.py` cobrindo `criar_db.py` e `main.py`, flags aceitas (`--base-dir`, `--prune`, `--session-id`), rejeição de flags obsoletas e códigos de saída 0 e != 0.
- [ ] **Task 8.4**: Refatorar `criar_db.py` como wrapper CLI aceitando exclusivamente argumentos `--base-dir` e `--prune` (removidas flags `--chunk-size/--chunk-overlap`), invocando `src.indexing.vector_store.VectorStoreManager.sync_directory()` e retornando código de saída `0` em sucesso e código != 0 em erro.
- [ ] **Task 8.5**: Refatorar `main.py` como wrapper CLI aceitando `--session-id`, invocando `src.cli.chat_loop.run_chat_loop()` e retornando código de saída `0` em encerramento normal.
- [ ] **Task 8.6**: Executar a suíte completa de testes unitários offline (`pytest -m "not eval" -v`) e análise de cobertura (`pytest --cov=src`), confirmando 100% de aprovação offline e cobertura superior a 85%.
