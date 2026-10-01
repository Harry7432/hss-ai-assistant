import hashlib
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import get_settings


class DocumentSplitter:
    """Divisor inteligente de documentos com hashing SHA-256 e IDs determinísticos."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def _compute_hash(self, text: str) -> str:
        """Calcula o hash SHA-256 de uma string de texto."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos em fragmentos com metadados enriquecidos e IDs determinísticos.

        Args:
            documents: Lista de documentos a serem fatiados.

        Returns:
            Lista de chunks com identificadores determinísticos {doc_hash[:12]}_{idx}.
        """
        raw_chunks = self.text_splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        # Mapeia índice sequencial por documento/arquivo
        doc_counters: dict[str, int] = {}

        for chunk in raw_chunks:
            source = chunk.metadata.get("source", "default")
            file_name = chunk.metadata.get("file_name", "unknown")
            page = chunk.metadata.get("page", 1)

            # Hash baseado no conteúdo e origem do documento
            doc_identifier = f"{file_name}_{page}"
            current_index = doc_counters.get(doc_identifier, 0)
            doc_counters[doc_identifier] = current_index + 1

            # Hash determinístico do conteúdo do chunk + fonte
            content_hash = self._compute_hash(chunk.page_content + "_" + file_name)

            meta = dict(chunk.metadata)
            meta["doc_hash"] = content_hash
            meta["chunk_index"] = current_index
            meta["chunk_id"] = f"{content_hash[:12]}_{current_index}"

            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=meta)
            )

        return enriched_chunks
