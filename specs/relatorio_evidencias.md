# Relatório Técnico de Evidências do Código Existente

**Projeto**: `hss-ai-assistant`  
**Data da Emissão**: 01/10/2026  
**Ambiente de Execução**: Windows (PowerShell), Python 3.13.15, Pytest 9.1.1, Coverage 7.1.0 (`.venv`)  
**Commit de Referência**: `13cce01` (`docs(specs): ajustar documentacao com os 7 itens calibrados`)  
**Status dos Arquivos Rastreados**: Intactos (nenhum arquivo de código ou teste foi modificado nesta inspeção)  

---

## 1. Sumário Executivo

Este relatório consolida a auditoria empírica e a coleta de evidências técnicas do código existente no projeto `hss-ai-assistant`. O objetivo é confrontar o estado real de implementação nos pacotes `src/`, `tests/` e scripts legados contra os requisitos funcionais, critérios de sucesso e os **7 itens de calibração técnica** estabelecidos nos documentos de especificação (`audit.md`, `spec.md`, `plan.md` e `tasks.md`).

### Principais Indicadores Encontrados

| Indicador | Valor Observado | Meta / Especificação Calibrada | Status |
| :--- | :--- | :--- | :--- |
| **Suíte de Testes Existente** | 30 testes coletados e executados | Testes unitários 100% offline | ✅ 30/30 Aprovados |
| **Tempo de Execução dos Testes** | 6.50s (puro) / 9.56s (com cobertura) | < 10.0s (SC-001) | ✅ Em conformidade (<10s) |
| **Custo de API em Testes Unitários** | R$ 0,00 (mocks offline) | 100% offline (NFR-001) | ✅ Em conformidade |
| **Cobertura de Código Atual** | **79%** (414 de 525 statements) | > 85% (NFR-005, SC-015) | ⚠️ Abaixo da meta (déficit de 6%) |
| **Módulos Críticos sem Testes** | `src/cli/chat_loop.py` (0%), `criar_db.py`, `main.py` | 100% dos adaptadores testados (Fase 8) | ⚠️ Pendente |
| **Alinhamento aos 7 Itens Técnicos** | Parcial (código reflete protótipo modular inicial v1.0) | Conformidade estrita v2.1 calibrada | ⚠️ Gaps identificados |

---

## 2. Evidências de Execução de Testes e Cobertura

A suíte foi executada através do interpretador isolado do ambiente virtual local (`.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing`).

### 2.1. Saída Bruta da Execução

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\harry\OneDrive\Área de Trabalho\Projetos\hss-ai-assistant
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, langsmith-0.14.2, cov-7.1.0, mock-3.16.0
collected 30 items

tests/eval/test_rag_quality.py::test_rag_quality_benchmarks PASSED       [  3%]
tests/integration/test_persistence.py::test_chroma_persistence_and_orphan_cleanup PASSED [  6%]
tests/unit/test_citations.py::test_format_citations_list PASSED          [ 10%]
tests/unit/test_citations.py::test_validate_citations_in_response PASSED [ 13%]
tests/unit/test_config.py::test_settings_default_values PASSED           [ 16%]
tests/unit/test_config.py::test_settings_custom_values PASSED            [ 20%]
tests/unit/test_config.py::test_settings_validation_invalid_chunk_overlap PASSED [ 23%]
tests/unit/test_config.py::test_settings_validation_invalid_threshold PASSED [ 26%]
tests/unit/test_config.py::test_get_settings_singleton PASSED            [ 30%]
tests/unit/test_guardrails.py::test_sanitize_log_message_masks_api_keys PASSED [ 33%]
tests/unit/test_guardrails.py::test_rag_engine_fallback_when_no_chunks PASSED [ 36%]
tests/unit/test_guardrails.py::test_rag_engine_retries_on_transient_error PASSED [ 40%]
tests/unit/test_loader.py::test_load_directory_nonexistent PASSED        [ 43%]
tests/unit/test_loader.py::test_load_documents_success PASSED            [ 46%]
tests/unit/test_loader.py::test_load_documents_empty_directory PASSED    [ 50%]
tests/unit/test_memory.py::test_session_store_add_and_retrieve PASSED    [ 53%]
tests/unit/test_memory.py::test_session_store_segregation PASSED         [ 56%]
tests/unit/test_memory.py::test_session_store_sliding_window PASSED      [ 60%]
tests/unit/test_prompt_templates.py::test_build_rag_prompt_structure_and_context_delimiters PASSED [ 63%]
tests/unit/test_recontextualizer.py::test_recontextualizer_empty_history PASSED [ 66%]
tests/unit/test_recontextualizer.py::test_recontextualizer_with_history PASSED [ 70%]
tests/unit/test_retrieval.py::test_retriever_search_ordering_and_metadata PASSED [ 73%]
tests/unit/test_retrieval.py::test_retriever_empty_collection PASSED     [ 76%]
tests/unit/test_retrieval.py::test_retriever_threshold_filtering PASSED  [ 80%]
tests/unit/test_splitter.py::test_split_documents_chunks_and_metadata PASSED [ 83%]
tests/unit/test_splitter.py::test_split_documents_deterministic_ids PASSED [ 86%]
tests/unit/test_vector_store.py::test_vector_store_initialization PASSED [ 90%]
tests/unit/test_vector_store.py::test_sync_documents_new_file PASSED     [ 93%]
tests/unit/test_vector_store.py::test_sync_documents_idempotence PASSED  [ 96%]
tests/unit/test_vector_store.py::test_sync_documents_removes_orphans PASSED [100%]

============================== warnings summary ===============================
tests\conftest.py:5: DeprecationWarning: `langchain-community` is being sunset...
tests/unit/test_retrieval.py::test_retriever_search_ordering_and_metadata: UserWarning: Relevance scores must be between 0 and 1...
tests/unit/test_retrieval.py::test_retriever_threshold_filtering: UserWarning: Relevance scores must be between 0 and 1...
======================= 30 passed, 3 warnings in 9.56s ========================
```

### 2.2. Relatório de Cobertura Detalhada por Módulo

| Módulo / Pacote | Statements | Miss | Cobertura | Linhas Faltantes |
| :--- | :---: | :---: | :---: | :--- |
| `src/__init__.py` | 1 | 0 | **100%** | - |
| `src/cli/__init__.py` | 2 | 2 | **0%** | Linhas 2-4 |
| `src/cli/chat_loop.py` | 49 | 49 | **0%** | Linhas 1-63 (sem testes unitários) |
| `src/core/__init__.py` | 3 | 0 | **100%** | - |
| `src/core/config.py` | 53 | 3 | **94%** | Linhas 57, 64, 78 |
| `src/core/exceptions.py` | 16 | 0 | **100%** | - |
| `src/generation/__init__.py` | 5 | 0 | **100%** | - |
| `src/generation/citations.py` | 37 | 1 | **97%** | Linha 47 |
| `src/generation/prompt_templates.py` | 12 | 0 | **100%** | - |
| `src/generation/rag_engine.py` | 68 | 3 | **96%** | Linhas 51, 75, 125 |
| `src/generation/recontextualizer.py` | 31 | 4 | **87%** | Linhas 32, 61-65 |
| `src/indexing/__init__.py` | 4 | 0 | **100%** | - |
| `src/indexing/loader.py` | 31 | 3 | **90%** | Linhas 33-35 |
| `src/indexing/splitter.py` | 31 | 0 | **100%** | - |
| `src/indexing/vector_store.py` | 79 | 18 | **77%** | Linhas 33, 48-50, 57-59, 94, 99-100, 103, 114, 129-134 |
| `src/memory/__init__.py` | 2 | 0 | **100%** | - |
| `src/memory/session_store.py` | 15 | 2 | **87%** | Linhas 26-27 |
| `src/observability/__init__.py` | 3 | 0 | **100%** | - |
| `src/observability/logging.py` | 24 | 11 | **54%** | Linhas 21, 32-33, 38-47 |
| `src/observability/telemetry.py` | 15 | 9 | **40%** | Linhas 15-25 (sem testes de opt-in/opt-out) |
| `src/retrieval/__init__.py` | 2 | 0 | **100%** | - |
| `src/retrieval/search.py` | 42 | 6 | **86%** | Linhas 54, 73-75, 90-93 |
| **TOTAL** | **525** | **111** | **79%** | **Meta: > 85%** |

---

## 3. Auditoria Componente a Componente e Confronto com Especificação

### 3.1. Configuração e Variáveis (`src/core/config.py` e `.env.example`)
- **Implementado**:
  - Modelo `Settings` baseado em Pydantic com validação de `chunk_size > 0`, `chunk_overlap >= 0` e `chunk_overlap < chunk_size`.
  - Leitura de variáveis do `.env` e singleton `get_settings()`.
- **Divergências com a Especificação Calibrada (v2.1)**:
  - `relevance_threshold` está com valor padrão `0.7` no código e no `.env.example`. A especificação calibrada para distância cosseno estipulou `0.35` como padrão inicial provisório calibrável via Task 3.3.
  - A variável `MAX_PRUNE_PERCENTAGE` (padrão 20%) exigida por FR-001/FR-005 não está definida em `Settings` nem documentada no `.env.example`.

### 3.2. Carregador e Divisor (`src/indexing/loader.py` e `src/indexing/splitter.py`)
- **Implementado**:
  - `PDFLoader`: Normalização para páginas 1-indexed (`int(raw_page) + 1`), captura de `file_name` e `source`.
  - `DocumentSplitter`: Chunking com `RecursiveCharacterTextSplitter`.
- **Divergências com a Especificação Calibrada**:
  - `splitter.py` (linhas 58-63): Calcula o hash SHA-256 sobre o fragmento individual (`content_hash = self._compute_hash(chunk.page_content + "_" + file_name)`) e trunca para 12 caracteres (`chunk_id = f"{content_hash[:12]}_{current_index}"`).
  - **Especificação FR-004**: Exige o hash SHA-256 completo do **arquivo** (`doc_hash`, 64 caracteres hexadecimais), sem cálculo redundante de hash por chunk individual, compondo `chunk_id = f"{doc_hash}_{chunk_index}"`.

### 3.3. Gerenciador do Banco Vetorial (`src/indexing/vector_store.py`)
- **Implementado**:
  - Sincronização básica com detecção de arquivos existentes vs novos, remoção de órfãos simples via `self.vector_store.delete(where={"file_name": orphan})`.
- **Divergências com a Especificação Calibrada (Item 5 da Calibração)**:
  - **Ausência de `manifest.json`**: O código atual não gera nem persiste arquivo de manifesto em disco; a checagem é feita por consulta dinâmica de metadados no ChromaDB.
  - **Ausência de Fingerprint da Configuração**: Não há verificação de fingerprint `{"embedding_model", "chunk_size", "chunk_overlap", "splitter_version", "distance_metric"}` nem provisionamento de coleção nova com nome derivado.
  - **Gravação Atômica Inexistente**: O manifesto não é gravado via `tmp` + rename.
  - **Reconciliação Inicial Inexistente**: Não há reconciliação manifesto x ChromaDB na inicialização.
  - **Rename vs Cópia**: O código atual apaga e reindexa se os hashes mudarem; não diferencia renomeação (atualizar metadados sem chamar embeddings) de cópias (warning e ignorar).
  - **Trava de Poda `--prune`**: Não existe proteção de 20% do índice (`MAX_PRUNE_PERCENTAGE`). Qualquer quantidade de arquivos ausentes é excluída cegamente.

### 3.4. Recuperação Vetorial (`src/retrieval/search.py`)
- **Implementado**:
  - `VectorRetriever`: Retorna lista tipada de `RetrievedChunk`, preserva metadados e trata coleção vazia com retorno estruturado de lista vazia.
- **Divergências com a Especificação Calibrada (Itens 1 e 7 da Calibração)**:
  - **Métrica e Direção do Score**:
    - O código utiliza `self.vector_store.similarity_search_with_relevance_scores()`. Essa função assume similaridade no intervalo $[0, 1]$ (maior é melhor). No ChromaDB com cosseno bruto, isso gerou o `UserWarning: Relevance scores must be between 0 and 1, got [... -159.07 ...]`.
    - O código ordena decrescentemente (`reverse=True`) e descarta com `if score < min_threshold:`.
    - **Especificação Calibrada (FR-006)**: A coleção DEVE ser configurada explicitamente com `configuration={"hnsw": {"space": "cosine"}}` (confirmado para ChromaDB 1.5.9), onde `score = distância` (MENOR é melhor), ordenação crescente e filtro `score <= RELEVANCE_THRESHOLD`.

### 3.5. Geração, Citações e Templates (`src/generation/`)
- **Implementado**:
  - `prompt_templates.py`: System prompt isolando `<context>` e regras para tratar contexto como dados passivos.
  - `citations.py`: Detecção regex de marcadores `[N]`, construção de snippets truncados em 140 caracteres.
  - `rag_engine.py`: Orquestrador completo integrando recontextualizador, memória, busca e geração.
- **Divergências com a Especificação Calibrada (Itens 3 e 6 da Calibração)**:
  - **Escape de Tags (FR-008)**: `prompt_templates.py` apenas faz `chunk.content.strip()`, sem escapar ocorrências de `</context>` e `<context>` inseridas no texto dos documentos.
  - **Frase-Sentinela**: Não há frase-sentinela fixa explícita no prompt com detecção automática para chavear `is_fallback=True`.
  - **Filtragem Estrita de Citações (FR-009)**: `CitationFormatter.format_sources_block` renderiza todos os chunks recuperados, e não exclusivamente os que foram citados no texto da resposta. Se restarem zero citações válidas, `rag_engine.py` não rebaixa a resposta para fallback.
  - **Retries e Mapeamento de Cota (FR-011)**: `rag_engine.py` utiliza um loop `for attempt in range(...)` manual com `time.sleep()`, capturando `Exception` genérico e lançando `GenerationError`. A especificação exige retries nativos (`max_retries=3`) no cliente OpenAI e mapeamento final de `insufficient_quota` para `QuotaExceededError`.

### 3.6. Observabilidade e CLI (`src/observability/`, `src/cli/`, wrappers)
- **Implementado**:
  - `logging.py`: Sanitizador regex para mascaramento de chaves `sk-...` e Bearer tokens.
  - `chat_loop.py`: Loop interativo com comandos `sair`, `limpar`, `sessao`.
  - `criar_db.py` e `main.py`: Wrappers legados chamando `VectorStoreManager` e `chat_loop`.
- **Divergências com a Especificação Calibrada (Fase 8 e Itens de Segurança)**:
  - `logging.py` grava logs em texto simples, em vez de JSON estruturado omitindo conteúdo textual de documentos por padrão. Não cobre explicitamente o padrão `sk-proj-...`.
  - `chat_loop.py` e `main.py` não realizam parsing de argumentos de linha de comando (`--session-id`).
  - `criar_db.py` não realiza parsing de `--base-dir` e `--prune`.

---

## 4. Confronto com os 7 Itens Técnicos Calibrados

| Item de Calibração | Estado no Código Atual | Diagnóstico / Evidência |
| :--- | :--- | :--- |
| **1. Calibração de Threshold (Task 3.3)** | ❌ Não implementado | Não existe o script `scripts/calibrate_threshold.py` nem rotina de calibração retrieval-only separada de `eval_dataset.json`. Threshold fixado em `0.7` no config. |
| **2. Eval: Regra Única de Regressão** | ⚠️ Parcial / Dummy | `tests/eval/test_rag_quality.py` possui apenas 4 casos mockados com `assert correct_rejections == 2` fixo. Não há portão por baseline gravado (`eval_baseline.json`) nem regra de perda de > N amostras sob temperatura 0. |
| **3. Testes de Cota/Retry (SC-001)** | ⚠️ Parcial | `test_guardrails.py` testa retry mockando `mock_llm.invoke.side_effect`, mas não há teste com transporte HTTP simulado nem verificação do mapeamento para `QuotaExceededError` com `max_retries=0` ou sleep mockado. |
| **4. Ordem TDD na Implementação** | ⚠️ Desalinhado no histórico | Os módulos de `src/` foram codificados previamente em bloco, resultando em módulos sem suítes de teste dedicadas (`test_chat_loop.py`, `test_cli_wrappers.py`, `test_telemetry.py`). |
| **5. Coleção Antiga vs Nova no Manifesto** | ❌ Não implementado | O gerenciador `vector_store.py` não implementa o ciclo do manifesto, fingerprint sem `collection_name` duplicado, ativação na gravação atômica nem remoção antiga sob flag `--prune`. |
| **6. Fixture Envenenada (Canário fora de `db/`)** | ❌ Não implementado | `eval_dataset.json` não contém documento-fixture envenenado com palavra-canário nem testes indexando estritamente em `tmp_path`. |
| **7. Padronização Task 3.3 (sem "Fase 3.5")** | ✅ Documentação 100% alinhada | Nos documentos de especificação, todas as referências foram unificadas para Task 3.3. No código, a Task 3.3 aguarda execução. |

---

## 5. Mapeamento de Status das Tarefas (`tasks.md`)

Conforme a regra de integridade de `tasks.md`, as tarefas permanecem formalmente como pendentes (`[ ]`) até que os testes dedicados e a calibração de cada fase sejam integralmente satisfeitos:

| Fase | Descrição | Status no Código | Evidência / Observação |
| :---: | :--- | :---: | :--- |
| **Fase 1** | Configurações, Exceções e Fixtures Base | **90%** | `config.py` e `exceptions.py` funcionais (testes passando). Falta campo `max_prune_percentage` e calibrar threshold padrão para 0.35. |
| **Fase 2** | Ingestão, Manifesto SHA-256 e Idempotência | **60%** | `loader.py` e `splitter.py` funcionais. Falta hash completo por arquivo (`doc_hash`), manifesto atômico, fingerprint e trava `--prune`. |
| **Fase 3** | Recuperação por Distância Cosseno e Calibração | **50%** | Busca funcional mas usando similaridade decrescente com warning de score. Falta distância cosseno explícita (1.5.9) e rotina de calibração (Task 3.3). |
| **Fase 4** | Geração com Citações e Templates | **70%** | Geração e citações operacionais. Falta escape de tags `<context>` na montagem, frase-sentinela e fontes citadas exclusivas. |
| **Fase 5** | Guardrails, Resiliência e Fallback Estrito | **60%** | Fallback operacional. Falta retries nativos OpenAI, mapeamento HTTP para `QuotaExceededError`, canário e logger JSON estruturado. |
| **Fase 6** | Memória Conversacional e Recontextualizador | **85%** | `session_store.py` e `recontextualizer.py` funcionais com testes unitários passando. |
| **Fase 7** | Observabilidade Segura e RAG Eval | **25%** | Telemetria com 40% de cobertura. Faltam `test_telemetry.py`, `test_performance.py` (SC-014) e dataset de 25+ amostras com fixture envenenada. |
| **Fase 8** | Adaptadores Legados e Validação Final | **30%** | Wrappers e CLI funcionam em nível básico. Faltam `test_chat_loop.py`, `test_cli_wrappers.py`, flags CLI e cobertura geral > 85%. |

---

## 6. Conclusões e Próximos Passos Sugeridos

1. **Base Sólida, Porém Pré-Calibração**: O repositório já dispõe de uma arquitetura modular funcional desacoplada e 30 testes unitários automatizados passando em menos de 10 segundos, o que comprova que os princípios fundamentais de modularização e mocks offline estão estabelecidos.
2. **Débitos Técnicos Principais para Próxima Etapa**:
   - Ajustar `splitter.py` para hash SHA-256 completo por arquivo (`doc_hash`) e IDs `{doc_hash}_{chunk_index}`.
   - Implementar `manifest.json` atômico com fingerprint e controle de nova coleção em `vector_store.py`.
   - Adequar `search.py` para distância cosseno real com ordenação crescente (`score <= RELEVANCE_THRESHOLD`).
   - Implementar testes unitários faltantes (`test_telemetry.py`, `test_chat_loop.py`, `test_cli_wrappers.py`, `test_performance.py`) para elevar a cobertura de **79%** para **> 85%**.
   - Criar o script da Task 3.3 para calibração empírica do threshold antes de fechar os guardrails.
