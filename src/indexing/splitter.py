from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import get_settings

SPLITTER_VERSION = "1"


def compute_file_hash(source: Path | str, fallback_content: str = "") -> str:
    """Calcula o hash SHA-256 completo lendo os bytes do arquivo em source.
    
    Caso o arquivo não exista no disco (documentos em memória ou testes sintéticos),
    utiliza o conteúdo e a origem como fallback.
    """
    if source:
        try:
            path = Path(source)
            if path.is_file():
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    while byte_block := f.read(65536):
                        hasher.update(byte_block)
                return hasher.hexdigest()
        except Exception:
            pass
    return hashlib.sha256((str(source) + "_" + fallback_content).encode("utf-8")).hexdigest()


class DocumentSplitter:
    """Divisor de documentos com hashing SHA-256 do arquivo e IDs determinísticos."""

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

    def _compute_file_hash(self, source: str, fallback_content: str = "") -> str:
        return compute_file_hash(source, fallback_content)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos em fragmentos com metadados enriquecidos e IDs determinísticos.

        Args:
            documents: Lista de documentos a serem fatiados.

        Returns:
            Lista de chunks com identificadores determinísticos {doc_hash}_{chunk_index}.
        """
        raw_chunks = self.text_splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        # Mapeia hash e contador sequencial de chunk por documento
        doc_hashes: dict[str, str] = {}
        doc_counters: dict[str, int] = {}
        indexed_now = datetime.now(timezone.utc).isoformat()

        for chunk in raw_chunks:
            source = chunk.metadata.get("source", "")
            file_name = chunk.metadata.get("file_name") or (Path(source).name if source else "unknown")
            page = chunk.metadata.get("page", 1)

            doc_key = str(source) if source else file_name
            if doc_key not in doc_hashes:
                doc_hashes[doc_key] = self._compute_file_hash(source, fallback_content=chunk.page_content)

            doc_hash = doc_hashes[doc_key]

            chunk_index = doc_counters.get(doc_key, 0)
            doc_counters[doc_key] = chunk_index + 1

            meta = dict(chunk.metadata)
            meta["file_name"] = file_name
            meta["source"] = str(source)
            meta["page"] = page
            meta["chunk_index"] = chunk_index
            meta["doc_hash"] = doc_hash
            meta["chunk_id"] = f"{doc_hash}_{chunk_index}"
            meta["indexed_at"] = indexed_now

            # Remove qualquer hash por chunk
            meta.pop("content_hash", None)
            meta.pop("chunk_hash", None)

            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=meta)
            )

        return enriched_chunks
