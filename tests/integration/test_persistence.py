import pytest
from pathlib import Path
from unittest.mock import patch
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.indexing.vector_store import VectorStoreManager
from src.core.config import Settings


def test_chroma_persistence_and_orphan_cleanup(tmp_path, mock_embeddings):
    """Valida ciclo completo de gravação em disco, idempotência e remoção de órfãos."""
    db_dir = tmp_path / "chroma_data"
    db_dir.mkdir()

    # Cria coleção com persistência real em disco temporário
    vector_store = Chroma(
        persist_directory=str(db_dir),
        embedding_function=mock_embeddings,
        collection_name="integration_test_col"
    )

    settings = Settings(
        chroma_persist_directory=str(db_dir),
        chunk_size=500,
        chunk_overlap=100
    )

    manager = VectorStoreManager(
        vector_store=vector_store,
        embeddings=mock_embeddings,
        settings=settings
    )

    doc_a = Document(
        page_content="Texto de integração do arquivo A sobre desenvolvimento de software.",
        metadata={"file_name": "arquivo_a.pdf", "page": 1, "source": "base/arquivo_a.pdf"}
    )
    doc_b = Document(
        page_content="Texto de integração do arquivo B com detalhes de infraestrutura.",
        metadata={"file_name": "arquivo_b.pdf", "page": 1, "source": "base/arquivo_b.pdf"}
    )

    # 1. Ingestão inicial de A e B
    with patch.object(manager.loader, "load_documents", return_value=[doc_a, doc_b]):
        res1 = manager.sync_directory("base")
        assert res1["added"] == 2
        assert manager.get_total_vectors() == 2

    # 2. Re-execução (idempotência)
    with patch.object(manager.loader, "load_documents", return_value=[doc_a, doc_b]):
        res2 = manager.sync_directory("base")
        assert res2["added"] == 0
        assert res2["skipped"] == 2
        assert manager.get_total_vectors() == 2

    # 3. Exclusão de arquivo_b (remoção de órfãos)
    with patch.object(manager.loader, "load_documents", return_value=[doc_a]):
        res3 = manager.sync_directory("base")
        assert res3["removed"] == 1
        assert res3["skipped"] == 1
        assert manager.get_total_vectors() == 1
