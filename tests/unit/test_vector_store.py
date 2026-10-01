import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.indexing.vector_store import VectorStoreManager
from src.core.exceptions import VectorStoreError


def test_vector_store_initialization(in_memory_chroma, mock_embeddings):
    """Valida inicialização do gerenciador com ChromaDB em memória."""
    manager = VectorStoreManager(vector_store=in_memory_chroma, embeddings=mock_embeddings)
    assert manager is not None
    assert manager.get_total_vectors() == 0


def test_sync_documents_new_file(in_memory_chroma, mock_embeddings):
    """Valida ingestão de arquivo inédito."""
    manager = VectorStoreManager(vector_store=in_memory_chroma, embeddings=mock_embeddings)

    doc = Document(
        page_content="Texto para indexação inicial.",
        metadata={"file_name": "guia.pdf", "page": 1, "source": "base/guia.pdf"}
    )

    with patch.object(manager.loader, "load_documents", return_value=[doc]):
        result = manager.sync_directory("base")

        assert result["added"] > 0
        assert manager.get_total_vectors() > 0


def test_sync_documents_idempotence(in_memory_chroma, mock_embeddings):
    """Valida que reexecução sobre arquivos inalterados não duplica vetores (idempotência)."""
    manager = VectorStoreManager(vector_store=in_memory_chroma, embeddings=mock_embeddings)

    doc = Document(
        page_content="Texto estático que não muda.",
        metadata={"file_name": "guia.pdf", "page": 1, "source": "base/guia.pdf"}
    )

    with patch.object(manager.loader, "load_documents", return_value=[doc]):
        # Primeira execução: adiciona
        res1 = manager.sync_directory("base")
        total_after_first = manager.get_total_vectors()

        # Segunda execução: deve ignorar (0 adicionados)
        res2 = manager.sync_directory("base")
        total_after_second = manager.get_total_vectors()

        assert res1["added"] > 0
        assert res2["added"] == 0
        assert res2["skipped"] > 0
        assert total_after_first == total_after_second


def test_sync_documents_removes_orphans(in_memory_chroma, mock_embeddings):
    """Valida que a remoção de um arquivo no disco limpa seus vetores do ChromaDB."""
    manager = VectorStoreManager(vector_store=in_memory_chroma, embeddings=mock_embeddings)

    doc1 = Document(
        page_content="Documento 1 que será mantido.",
        metadata={"file_name": "doc1.pdf", "page": 1, "source": "base/doc1.pdf"}
    )
    doc2 = Document(
        page_content="Documento 2 que será excluído posteriormente.",
        metadata={"file_name": "doc2.pdf", "page": 1, "source": "base/doc2.pdf"}
    )

    # 1. Indexar doc1 e doc2
    with patch.object(manager.loader, "load_documents", return_value=[doc1, doc2]):
        manager.sync_directory("base")
        initial_count = manager.get_total_vectors()
        assert initial_count >= 2

    # 2. Sincronizar apenas com doc1 (doc2 foi excluído do diretório)
    with patch.object(manager.loader, "load_documents", return_value=[doc1]):
        res = manager.sync_directory("base")
        assert res["removed"] > 0
        new_count = manager.get_total_vectors()
        assert new_count < initial_count
