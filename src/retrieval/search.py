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

    @property
    def file_name(self) -> str:
        return str(self.metadata.get("file_name", ""))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 1))

    @property
    def doc_hash(self) -> str:
        return str(self.metadata.get("doc_hash", ""))


class VectorRetriever:
    """Recuperador semântico sobre ChromaDB com ordenação por distância cosseno."""

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
        """Realiza busca semântica por distância cosseno.

        IMPORTANTE: O score retornado é a distância cosseno geométrica bruta
        no embedding space (menor é melhor, 0 = idêntico), e NÃO uma probabilidade
        bayesiana de certeza factual ou score normalizado.

        Args:
            query: Texto da consulta.
            k: Número máximo de resultados a recuperar (padrão: settings.retrieval_k).
            threshold: Distância máxima aceita (padrão: settings.relevance_threshold).

        Returns:
            Lista de RetrievedChunk ordenados crescentemente por distância (menor score primeiro).
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        limit_k = k if k is not None else self.settings.retrieval_k
        max_distance = (
            threshold if threshold is not None else self.settings.relevance_threshold
        )

        try:
            # Verifica se o banco tem documentos cadastrados
            total_docs = self.vector_store._collection.count()
            if total_docs == 0:
                logger.debug("Busca em coleção vazia; retornando lista vazia.")
                return []

            raw_results = self.vector_store.similarity_search_with_score(
                clean_query, k=limit_k
            )
        except Exception as e:
            logger.error(f"Erro ao executar busca de similaridade no ChromaDB: {e}")
            raise RetrievalError(f"Falha na busca vetorial: {e}") from e

        # Ordenar os resultados por distância em ordem crescente (menor distância = maior similaridade)
        sorted_results = sorted(raw_results, key=lambda item: item[1])

        retrieved_chunks: List[RetrievedChunk] = []

        for doc, distance in sorted_results:
            score = float(distance)
            # Filtragem estrita: score menor ou igual a RELEVANCE_THRESHOLD
            if score > max_distance:
                logger.debug(
                    f"Chunk descartado: distância {score:.4f} excede threshold {max_distance}"
                )
                continue

            metadata = dict(doc.metadata) if doc.metadata else {}
            chunk_id = metadata.get(
                "chunk_id", f"chk_{abs(hash(doc.page_content))}"
            )
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = chunk_id

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    metadata=metadata,
                    score=score,
                )
            )

        logger.debug(
            f"Busca '{clean_query}': {len(retrieved_chunks)}/{len(raw_results)} chunks aceitos."
        )
        return retrieved_chunks
