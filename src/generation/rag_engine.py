import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from src.core.config import Settings, get_settings
from src.core.exceptions import GenerationError
from src.generation.citations import Citation, CitationFormatter
from src.generation.prompt_templates import build_rag_prompt
from src.generation.recontextualizer import QueryRecontextualizer
from src.memory.session_store import SessionStore
from src.observability.logging import sanitize_log_message
from src.retrieval.search import RetrievedChunk, VectorRetriever

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "Não foi possível encontrar informações suficientes na base de conhecimento "
    "para responder à sua pergunta com segurança."
)


@dataclass
class RAGResponse:
    """Resposta estruturada gerada pelo motor RAG."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    is_fallback: bool = False
    session_id: str = "default"
    metrics: Dict[str, Any] = field(default_factory=dict)


class RAGEngine:
    """Orquestrador central do pipeline RAG modular com guardrails e resiliência."""

    def __init__(
        self,
        vector_store: Chroma,
        llm: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.retriever = VectorRetriever(vector_store=self.vector_store, settings=self.settings)

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.0,
                openai_api_key=self.settings.openai_api_key,
            )

        self.recontextualizer = QueryRecontextualizer(llm=self.llm, settings=self.settings)
        self.session_store = SessionStore()
        self.citation_formatter = CitationFormatter()

    def query(self, question: str, session_id: str = "default") -> RAGResponse:
        """Executa o ciclo completo de consulta: memória -> reescrita -> busca -> geração -> validação.

        Args:
            question: Pergunta realizada pelo usuário.
            session_id: Identificador único da sessão para memória multi-turn.

        Returns:
            Instância de RAGResponse com texto, fontes, flags e métricas.
        """
        start_time = time.time()
        clean_question = question.strip() if question else ""

        if not clean_question:
            return RAGResponse(
                answer="Por favor, digite uma pergunta válida.",
                is_fallback=True,
                session_id=session_id,
                metrics={"latency_ms": 0.0},
            )

        # 1. Recuperar histórico da sessão e recontextualizar query
        history = self.session_store.get_history(session_id)
        search_query = self.recontextualizer.recontextualize(clean_question, history=history)

        # 2. Recuperação vetorial com ordenação e threshold
        chunks = self.retriever.search(search_query)

        # 3. Guardrail / Fallback estrito: ausência de contexto suficiente
        if not chunks:
            logger.info(
                sanitize_log_message(
                    f"Fallback acionado para a pergunta: '{clean_question}'. Nenhum chunk acima do threshold."
                )
            )
            elapsed_ms = (time.time() - start_time) * 1000
            return RAGResponse(
                answer=FALLBACK_MESSAGE,
                citations=[],
                is_fallback=True,
                session_id=session_id,
                metrics={"latency_ms": elapsed_ms, "chunks_count": 0},
            )

        # 4. Construção de prompt seguro e delimitado
        prompt = build_rag_prompt(query=clean_question, chunks=chunks)
        citations = self.citation_formatter.build_citations(chunks)
        valid_markers = [c.marker for c in citations]

        # 5. Execução do modelo LLM com política de retries
        raw_answer = ""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                raw_answer = getattr(response, "content", str(response)).strip()
                break
            except Exception as e:
                logger.warning(
                    sanitize_log_message(
                        f"Tentativa {attempt}/{max_retries} falhou ao invocar LLM: {e}"
                    )
                )
                if attempt == max_retries:
                    raise GenerationError(
                        f"Falha de comunicação persistente com o modelo de IA: {e}"
                    ) from e
                time.sleep(0.5 * (2 ** (attempt - 1)))

        # 6. Validação programática de citações geradas
        cleaned_answer, _ = self.citation_formatter.validate_response_citations(
            raw_answer, valid_markers
        )

        # Anexar bloco de fontes consultadas
        sources_block = self.citation_formatter.format_sources_block(citations)
        final_answer = cleaned_answer + sources_block

        # 7. Registrar turno na memória da sessão
        self.session_store.add_turn(
            session_id=session_id,
            user_message=clean_question,
            assistant_message=cleaned_answer,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            sanitize_log_message(
                f"Consulta concluída em {elapsed_ms:.2f}ms para sessão '{session_id}' ({len(chunks)} fontes)."
            )
        )

        return RAGResponse(
            answer=final_answer,
            citations=citations,
            is_fallback=False,
            session_id=session_id,
            metrics={"latency_ms": elapsed_ms, "chunks_count": len(chunks)},
        )
