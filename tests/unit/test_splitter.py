import hashlib
from datetime import datetime
from pathlib import Path
import pytest
from langchain_core.documents import Document
from src.indexing.splitter import DocumentSplitter
from src.core.config import get_settings


def test_splitter_settings_defaults():
    """Valida que chunk_size e chunk_overlap padrão vêm do Settings."""
    settings = get_settings()
    splitter = DocumentSplitter()
    assert splitter.chunk_size == settings.chunk_size
    assert splitter.chunk_overlap == settings.chunk_overlap


def test_splitter_custom_parameters():
    """Valida passagem de parâmetros customizados de chunk_size e chunk_overlap."""
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 100


def test_splitter_mandatory_metadata_and_no_chunk_hash(tmp_path):
    """Valida metadados obrigatórios, formato do chunk_id e ausência de hash por chunk."""
    test_file = tmp_path / "manual.pdf"
    content_bytes = b"PDF_CONTENT_TEST_BYTES"
    test_file.write_bytes(content_bytes)
    expected_doc_hash = hashlib.sha256(content_bytes).hexdigest()

    page_text = "Parágrafo 1 de conteúdo informativo. " * 30
    doc = Document(
        page_content=page_text,
        metadata={"file_name": "manual.pdf", "page": 1, "source": str(test_file)}
    )

    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1

    mandatory_keys = {"file_name", "source", "page", "chunk_index", "doc_hash", "chunk_id", "indexed_at"}
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        # Validação de todos os metadados obrigatórios
        for key in mandatory_keys:
            assert key in meta, f"Metadado obrigatório '{key}' ausente no chunk {idx}"

        assert meta["file_name"] == "manual.pdf"
        assert meta["source"] == str(test_file)
        assert meta["page"] == 1
        assert meta["chunk_index"] == idx
        assert meta["doc_hash"] == expected_doc_hash
        assert meta["chunk_id"] == f"{expected_doc_hash}_{idx}"

        # Validação de formato ISO-8601 de indexed_at
        datetime.fromisoformat(meta["indexed_at"])

        # Ausência de hash por chunk
        assert "content_hash" not in meta
        assert "chunk_hash" not in meta


def test_splitter_same_file_generates_same_ids(tmp_path):
    """Valida que o mesmo arquivo no disco gera exatamente os mesmos IDs determinísticos."""
    test_file = tmp_path / "mesmo_arquivo.pdf"
    test_file.write_bytes(b"CONTEUDO_IDENTICO_123")

    doc = Document(
        page_content="Texto de teste para validação de idempotência.",
        metadata={"file_name": "mesmo_arquivo.pdf", "page": 1, "source": str(test_file)}
    )

    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    chunks_run1 = splitter.split_documents([doc])
    chunks_run2 = splitter.split_documents([doc])

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.metadata["doc_hash"] == c2.metadata["doc_hash"]
        assert c1.metadata["chunk_id"] == c2.metadata["chunk_id"]
        assert c1.metadata["chunk_index"] == c2.metadata["chunk_index"]


def test_splitter_altered_file_generates_different_ids(tmp_path):
    """Valida que um arquivo alterado gera doc_hash e chunk_ids diferentes."""
    test_file = tmp_path / "arquivo_mutavel.pdf"
    test_file.write_bytes(b"CONTEUDO_ORIGINAL")

    doc = Document(
        page_content="Texto do documento que será alterado.",
        metadata={"file_name": "arquivo_mutavel.pdf", "page": 1, "source": str(test_file)}
    )

    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    chunks_original = splitter.split_documents([doc])

    # Modifica o conteúdo físico do arquivo
    test_file.write_bytes(b"CONTEUDO_ALTERADO_TOTALMENTE_DIFERENTE")
    chunks_altered = splitter.split_documents([doc])

    assert chunks_original[0].metadata["doc_hash"] != chunks_altered[0].metadata["doc_hash"]
    assert chunks_original[0].metadata["chunk_id"] != chunks_altered[0].metadata["chunk_id"]


def test_splitter_sequential_chunk_index_across_pages(tmp_path):
    """Valida que páginas do mesmo documento recebem chunk_index sequencial contínuo começando em 0."""
    test_file = tmp_path / "multi_paginas.pdf"
    test_file.write_bytes(b"BYTES_PDF_MULTIPAGINAS")

    docs = [
        Document(
            page_content="Pagina 1 " * 30,
            metadata={"file_name": "multi_paginas.pdf", "page": 1, "source": str(test_file)}
        ),
        Document(
            page_content="Pagina 2 " * 30,
            metadata={"file_name": "multi_paginas.pdf", "page": 2, "source": str(test_file)}
        ),
    ]

    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
    assert indices[0] == 0

    doc_hash = chunks[0].metadata["doc_hash"]
    for idx, c in enumerate(chunks):
        assert c.metadata["chunk_id"] == f"{doc_hash}_{idx}"
