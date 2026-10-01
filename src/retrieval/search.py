import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from langchain_chroma import Chroma
from src.core.config import Settings, get_settings
from src.core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Representação de um fragmento recuperado na busca vetorial."""

    chunk_id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    score: float = 0.0


class VectorRetriever:
    """Recuperador semântico sobre ChromaDB com ordenação e calibragem heurística."""

    def __init__(
        self,
        vector_store: Chroma,
        settings: Optional[Settings] = None,
    ):
        self.vector_store = vector_store
        self.settings = settings or get_settings()

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Realiza busca semântica por proximidade vetorial.

        IMPORTANTE: O score retornado é uma métrica heurística de distância/proximidade
        espacial no embedding space (cosseno/L2), e NÃO uma probabilidade bayesiana
        de certeza factual ou correção da resposta.

        Args:
            query: Texto da consulta.
            k: Número máximo de resultados a recuperar (padrão: settings.retrieval_k).
            threshold: Limiar mínimo de score (padrão: settings.relevance_threshold).

        Returns:
            Lista de RetrievedChunk ordenados decrescentemente por relevância.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        limit_k = k if k is not None else self.settings.retrieval_k
        min_threshold = (
            threshold if threshold is not None else self.settings.relevance_threshold
        )

        try:
            # Verifica se o banco tem documentos cadastrados
            total_docs = self.vector_store._collection.count()
            if total_docs == 0:
                logger.debug("Busca em coleção vazia; retornando lista vazia.")
                return []

            raw_results = (
                self.vector_store.similarity_search_with_relevance_scores(
                    clean_query, k=limit_k
                )
            )
        except Exception as e:
            logger.error(f"Erro ao executar busca de similaridade no ChromaDB: {e}")
            raise RetrievalError(f"Falha na busca vetorial: {e}") from e

        retrieved_chunks: List[RetrievedChunk] = []

        # Ordenar os resultados por score de forma decrescente
        sorted_results = sorted(raw_results, key=lambda item: item[1], reverse=True)

        for doc, score in sorted_results:
            # Filtragem estrita por limiar heurístico
            if score < min_threshold:
                logger.debug(
                    f"Chunk descartado: score {score:.4f} abaixo do threshold {min_threshold}"
                )
                continue

            chunk_id = doc.metadata.get(
                "chunk_id", f"chk_{abs(hash(doc.page_content))}"
            )
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    score=float(score),
                )
            )

        logger.debug(
            f"Busca '{clean_query}': {len(retrieved_chunks)}/{len(raw_results)} chunks aceitos."
        )
        return retrieved_chunks
