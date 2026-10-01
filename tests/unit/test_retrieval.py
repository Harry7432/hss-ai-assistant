import pytest
from langchain_core.documents import Document
from src.retrieval.search import VectorRetriever, RetrievedChunk
from src.core.config import Settings


def test_retriever_search_ordering_and_metadata(in_memory_chroma, mock_embeddings):
    """Valida busca semântica, ordenação por proximidade e integridade de metadados."""
    docs = [
        Document(
            page_content="Introdução à linguagem Python e sintaxe básica.",
            metadata={"file_name": "faq.pdf", "page": 1, "chunk_id": "doc1_0"}
        ),
        Document(
            page_content="Estruturas de repetição: loops for e while em Python.",
            metadata={"file_name": "faq.pdf", "page": 2, "chunk_id": "doc1_1"}
        ),
        Document(
            page_content="Culinária mediterrânea e receitas tradicionais.",
            metadata={"file_name": "culinaria.pdf", "page": 1, "chunk_id": "doc2_0"}
        ),
    ]
    in_memory_chroma.add_documents(docs)

    retriever = VectorRetriever(vector_store=in_memory_chroma, settings=Settings(retrieval_k=2))
    results = retriever.search("Como usar loops em Python?")

    assert len(results) <= 2
    for item in results:
        assert isinstance(item, RetrievedChunk)
        assert item.content is not None
        assert "file_name" in item.metadata
        assert "page" in item.metadata
        # Score deve ser um valor numérico de métrica geométrica (não probabilidade bayesiana)
        assert isinstance(item.score, float)


def test_retriever_empty_collection(in_memory_chroma):
    """Valida comportamento seguro ao consultar uma base vetorial vazia."""
    retriever = VectorRetriever(vector_store=in_memory_chroma)
    results = retriever.search("Qualquer pergunta")
    assert results == []


def test_retriever_threshold_filtering(in_memory_chroma):
    """Valida que chunks com pontuação inferior ao limiar são descartados."""
    doc = Document(
        page_content="Informação específica sobre computação quântica.",
        metadata={"file_name": "fisica.pdf", "page": 10, "chunk_id": "doc3_0"}
    )
    in_memory_chroma.add_documents([doc])

    # Configurar threshold muito alto para forçar corte
    retriever = VectorRetriever(
        vector_store=in_memory_chroma,
        settings=Settings(relevance_threshold=0.999)
    )
    results = retriever.search("Receita de bolo de chocolate")
    assert len(results) == 0
