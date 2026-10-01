import shutil
from pathlib import Path
from unittest.mock import patch
import pytest
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from src.core.config import Settings
from src.indexing.loader import LoadResult
from src.indexing.manifest import IndexManifest
from src.indexing.vector_store import VectorStoreManager, SyncReport


def test_chroma_persistence_between_runs(tmp_path):
    """1. Persistência real entre execuções em disco com Chroma e manifesto."""
    db_dir = tmp_path / "chroma_db"
    docs_dir = tmp_path / "base"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"

    f1 = docs_dir / "arquivo_a.pdf"
    f1.write_bytes(b"%PDF-1.4 arquivo A bytes")

    doc_a = Document(
        page_content="Texto de integração do arquivo A sobre desenvolvimento de software.",
        metadata={"file_name": "arquivo_a.pdf", "page": 1, "source": str(f1)}
    )

    embeddings = DeterministicFakeEmbedding(size=128)
    settings = Settings(
        chroma_persist_directory=str(db_dir),
        documents_directory=str(docs_dir),
        collection_name="test_persistence_col",
    )

    # Execução 1: Ingestão inicial
    manager1 = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
        manifest_path=manifest_path,
    )

    with patch.object(manager1.loader, "load_documents", return_value=LoadResult(documents=[doc_a])):
        res1 = manager1.sync_directory()
        assert res1.indexed == 1
        assert manager1.get_total_vectors() > 0

    # Execução 2: Nova instância do gerenciador apontando para o mesmo diretório
    manager2 = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
        manifest_path=manifest_path,
    )

    with patch.object(manager2.loader, "load_documents", return_value=LoadResult(documents=[doc_a])):
        res2 = manager2.sync_directory()
        assert res2.unchanged == 1
        assert res2.indexed == 0
        assert res2.embedding_calls == 0


def test_chroma_wipe_directory_keeping_manifest_reindexes_all(tmp_path):
    """2. Apagar a pasta do Chroma mantendo o manifesto reindexa tudo."""
    db_dir = tmp_path / "chroma_db"
    docs_dir = tmp_path / "base"
    docs_dir.mkdir(parents=True, exist_ok=True)
    # Manifesto armazenado fora da pasta apagada para que sobreviva à limpeza
    manifest_path = tmp_path / "manifest_persisted.json"

    f1 = docs_dir / "guia.pdf"
    f1.write_bytes(b"%PDF-1.4 guia content")

    doc = Document(
        page_content="Guia completo de arquitetura.",
        metadata={"file_name": "guia.pdf", "page": 1, "source": str(f1)}
    )

    embeddings = DeterministicFakeEmbedding(size=128)
    settings = Settings(
        chroma_persist_directory=str(db_dir),
        documents_directory=str(docs_dir),
        collection_name="col_reindex_on_wipe",
    )

    # Execução 1: Ingestão inicial
    manager1 = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
        manifest_path=manifest_path,
    )

    with patch.object(manager1.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        res1 = manager1.sync_directory()
        assert res1.indexed == 1
        assert manifest_path.exists()

    # Apagar o diretório do ChromaDB (simula perda do banco mantendo manifesto)
    # Fechar conexões e liberar handles do SQLite no Windows
    if hasattr(manager1, "client") and manager1.client is not None:
        manager1.client.close()
        manager1.client.clear_system_cache()
    del manager1
    if db_dir.exists():
        shutil.rmtree(db_dir)

    # Execução 2: Novo gerenciador. O manifesto existe, mas o Chroma foi apagado.
    # A divergência (Chroma vazio vs manifesto com IDs) deve disparar REINDEXAÇÃO TOTAL!
    manager2 = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
        manifest_path=manifest_path,
    )

    with patch.object(manager2.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        res2 = manager2.sync_directory()
        # Constata reindexação total
        assert res2.indexed == 1
        assert manager2.get_total_vectors() > 0


def test_chroma_collection_cosine_configuration(tmp_path):
    """3. Lê collection.configuration e confirma distância cosseno."""
    db_dir = tmp_path / "chroma_db"
    settings = Settings(
        chroma_persist_directory=str(db_dir),
        collection_name="cosine_test_col",
    )
    embeddings = DeterministicFakeEmbedding(size=128)

    manager = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
    )

    # Inspeciona a configuração real da coleção Chroma
    col_config = manager.vector_store._collection.configuration
    assert col_config is not None
    assert "hnsw" in col_config
    assert col_config["hnsw"]["space"] == "cosine", (
        f"Esperava métrica 'cosine', obteve: {col_config['hnsw'].get('space')}"
    )
