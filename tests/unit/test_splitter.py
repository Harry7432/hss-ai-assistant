import hashlib
import pytest
from langchain_core.documents import Document
from src.indexing.splitter import DocumentSplitter
from src.core.config import Settings


def test_split_documents_chunks_and_metadata():
    """Valida fatiamento de documentos, metadados enriquecidos e identificadores determinísticos."""
    content = "A" * 800 + " " + "B" * 800
    doc = Document(
        page_content=content,
        metadata={"file_name": "manual.pdf", "page": 1, "source": "base/manual.pdf"}
    )

    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert "chunk_id" in chunk.metadata
        assert "doc_hash" in chunk.metadata
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["file_name"] == "manual.pdf"
        assert chunk.metadata["page"] == 1
        # Formato esperado: {doc_hash[:12]}_{chunk_index}
        doc_hash = chunk.metadata["doc_hash"]
        assert chunk.metadata["chunk_id"] == f"{doc_hash[:12]}_{idx}"


def test_split_documents_deterministic_ids():
    """Valida que o mesmo conteúdo gera os mesmos IDs de chunk (idempotência)."""
    doc = Document(
        page_content="Conteúdo imutável do documento de teste.",
        metadata={"file_name": "teste.pdf", "page": 1, "source": "base/teste.pdf"}
    )

    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)
    chunks1 = splitter.split_documents([doc])
    chunks2 = splitter.split_documents([doc])

    assert len(chunks1) == len(chunks2)
    assert chunks1[0].metadata["chunk_id"] == chunks2[0].metadata["chunk_id"]
    assert chunks1[0].metadata["doc_hash"] == chunks2[0].metadata["doc_hash"]
