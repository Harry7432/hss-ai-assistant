import os
import pytest
from typing import List
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_community.llms.fake import FakeListLLM
import chromadb
from langchain_chroma import Chroma


@pytest.fixture
def mock_embeddings():
    """Retorna uma classe de embedding simulada para testes rápidos offline."""
    return FakeEmbeddings(size=128)


@pytest.fixture
def mock_llm_factory():
    """Fábrica para instanciar FakeListLLM com respostas personalizadas."""
    def _create(responses: List[str] = None):
        if responses is None:
            responses = ["Esta é uma resposta de teste [1]."]
        return FakeListLLM(responses=responses)
    return _create


import uuid


@pytest.fixture
def in_memory_chroma(mock_embeddings):
    """Instancia um ChromaDB em memória isolado para testes unitários."""
    client = chromadb.EphemeralClient()
    collection_name = f"test_col_{uuid.uuid4().hex}"
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=mock_embeddings
    )
    return vector_store


@pytest.fixture
def temp_documents_dir(tmp_path):
    """Cria um diretório temporário para testes de ingestão de arquivos."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir
