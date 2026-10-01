import uuid
from typing import List
import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import Settings
from src.retrieval.search import RetrievedChunk, VectorRetriever


class ControlledEmbeddings(Embeddings):
    """Embeddings determinísticos com vetores controlados para testar distância cosseno no Chroma."""

    def __init__(self, mapping: dict[str, list[float]], default_vector: list[float] = None):
        self.mapping = mapping
        self.default_vector = default_vector or [0.0, 1.0]

    def _embed(self, text: str) -> list[float]:
        for key, vec in self.mapping.items():
            if key in text:
                return vec
        return self.default_vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def controlled_chroma():
    """Cria uma coleção Chroma com distância cosseno e ControlledEmbeddings calibrados."""
    mapping = {
        "query": [1.0, 0.0],
        "identico": [1.0, 0.0],
        "intermediario": [0.7071067811865475, 0.7071067811865475],
        "ortogonal": [0.0, 1.0],
    }
    embeddings = ControlledEmbeddings(mapping=mapping, default_vector=[0.0, 1.0])
    client = chromadb.EphemeralClient()
    col_name = f"test_controlled_{uuid.uuid4().hex}"
    client.get_or_create_collection(
        name=col_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
    vector_store = Chroma(
        client=client,
        collection_name=col_name,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    return vector_store


# ---------------------------------------------------------------------------
# Testes legados ajustados ao novo contrato (distância cosseno, menor = melhor)
# ---------------------------------------------------------------------------

def test_retriever_search_ordering_and_metadata(cosine_chroma, mock_embeddings):
    """Valida busca semântica, ordenação crescente por distância e integridade de metadados."""
    docs = [
        Document(
            page_content="Introdução à linguagem Python e sintaxe básica.",
            metadata={"file_name": "faq.pdf", "page": 1, "chunk_id": "doc1_0", "source": "base/faq.pdf", "doc_hash": "hash1"}
        ),
        Document(
            page_content="Estruturas de repetição: loops for e while em Python.",
            metadata={"file_name": "faq.pdf", "page": 2, "chunk_id": "doc1_1", "source": "base/faq.pdf", "doc_hash": "hash1"}
        ),
        Document(
            page_content="Culinária mediterrânea e receitas tradicionais.",
            metadata={"file_name": "culinaria.pdf", "page": 1, "chunk_id": "doc2_0", "source": "base/culinaria.pdf", "doc_hash": "hash2"}
        ),
    ]
    cosine_chroma.add_documents(docs)

    retriever = VectorRetriever(vector_store=cosine_chroma, settings=Settings(retrieval_k=2, relevance_threshold=1.0))
    results = retriever.search("Como usar loops em Python?")

    assert len(results) <= 2
    assert len(results) > 0
    # Ordem crescente de distância: menor score primeiro
    if len(results) > 1:
        assert results[0].score <= results[1].score

    for item in results:
        assert isinstance(item, RetrievedChunk)
        assert item.content is not None
        assert "file_name" in item.metadata
        assert "page" in item.metadata
        assert item.metadata["page"] >= 1  # 1-indexed
        # Score deve ser uma distância geométrica bruta não negativa
        assert isinstance(item.score, float)
        assert item.score >= 0.0


def test_retriever_empty_collection(cosine_chroma):
    """Valida comportamento seguro ao consultar uma base vetorial vazia."""
    retriever = VectorRetriever(vector_store=cosine_chroma)
    results = retriever.search("Qualquer pergunta")
    assert results == []


def test_retriever_threshold_filtering(cosine_chroma):
    """Valida que chunks com distância superior ao limiar são descartados."""
    doc = Document(
        page_content="Informação específica sobre computação quântica.",
        metadata={"file_name": "fisica.pdf", "page": 10, "chunk_id": "doc3_0", "source": "base/fisica.pdf", "doc_hash": "hash3"}
    )
    cosine_chroma.add_documents([doc])

    # Limiar muito rigoroso (distância máxima quase 0), forçando o descarte de documentos distantes
    retriever = VectorRetriever(
        vector_store=cosine_chroma,
        settings=Settings(relevance_threshold=0.0001)
    )
    results = retriever.search("Receita de bolo de chocolate")
    assert results == []


# ---------------------------------------------------------------------------
# Novos testes com ControlledEmbeddings (vetores calibrados em espaço cosseno)
# ---------------------------------------------------------------------------

def test_retriever_controlled_embeddings_ordering(controlled_chroma):
    """Valida ordenação estritamente crescente de distância (identico=0, intermediario~0.29, ortogonal~1.0)."""
    docs = [
        Document(
            page_content="Texto ortogonal ao vetor de busca.",
            metadata={"file_name": "orto.pdf", "page": 1, "chunk_id": "c_orto", "source": "orto.pdf", "doc_hash": "h_orto"}
        ),
        Document(
            page_content="Texto identico ao vetor de busca.",
            metadata={"file_name": "id.pdf", "page": 1, "chunk_id": "c_id", "source": "id.pdf", "doc_hash": "h_id"}
        ),
        Document(
            page_content="Texto intermediario ao vetor de busca.",
            metadata={"file_name": "inter.pdf", "page": 1, "chunk_id": "c_inter", "source": "inter.pdf", "doc_hash": "h_inter"}
        ),
    ]
    controlled_chroma.add_documents(docs)

    retriever = VectorRetriever(vector_store=controlled_chroma, settings=Settings(retrieval_k=3, relevance_threshold=1.0))
    results = retriever.search("query de busca")

    assert len(results) == 3
    # Ordem crescente de distância: menor score primeiro
    assert results[0].score < results[1].score < results[2].score
    assert "identico" in results[0].content
    assert "intermediario" in results[1].content
    assert "ortogonal" in results[2].content


def test_retriever_controlled_embeddings_threshold_filtering(controlled_chroma):
    """Valida filtragem por RELEVANCE_THRESHOLD (score <= threshold é mantido, score > threshold é descartado)."""
    docs = [
        Document(
            page_content="Texto identico ao vetor de busca.",
            metadata={"file_name": "id.pdf", "page": 1, "chunk_id": "c_id", "source": "id.pdf", "doc_hash": "h_id"}
        ),
        Document(
            page_content="Texto intermediario ao vetor de busca.",
            metadata={"file_name": "inter.pdf", "page": 1, "chunk_id": "c_inter", "source": "inter.pdf", "doc_hash": "h_inter"}
        ),
        Document(
            page_content="Texto ortogonal ao vetor de busca.",
            metadata={"file_name": "orto.pdf", "page": 1, "chunk_id": "c_orto", "source": "orto.pdf", "doc_hash": "h_orto"}
        ),
    ]
    controlled_chroma.add_documents(docs)

    # Threshold 0.35: identico (0.0 <= 0.35) e intermediario (~0.2929 <= 0.35) passam; ortogonal (~1.0 > 0.35) cai
    retriever = VectorRetriever(vector_store=controlled_chroma, settings=Settings(retrieval_k=3, relevance_threshold=0.35))
    results = retriever.search("query de busca")

    assert len(results) == 2
    assert all(r.score <= 0.35 for r in results)
    contents = [r.content for r in results]
    assert any("identico" in c for c in contents)
    assert any("intermediario" in c for c in contents)
    assert not any("ortogonal" in c for c in contents)


def test_retriever_controlled_embeddings_metadata_preservation_and_1_indexed(controlled_chroma):
    """Valida que metadados completos (file_name, source, page 1-indexed, chunk_id, doc_hash) e conteúdo são preservados."""
    doc = Document(
        page_content="Conteúdo original intacto sem escape.",
        metadata={
            "file_name": "manual_tecnico.pdf",
            "source": "base/manual_tecnico.pdf",
            "page": 3,
            "chunk_id": "hash999_2",
            "doc_hash": "hash999",
        }
    )
    controlled_chroma.add_documents([doc])

    retriever = VectorRetriever(vector_store=controlled_chroma, settings=Settings(retrieval_k=1, relevance_threshold=1.0))
    results = retriever.search("query qualquer")

    assert len(results) == 1
    chunk = results[0]
    assert chunk.content == "Conteúdo original intacto sem escape."
    assert chunk.metadata["file_name"] == "manual_tecnico.pdf"
    assert chunk.metadata["source"] == "base/manual_tecnico.pdf"
    assert chunk.metadata["page"] == 3
    assert chunk.metadata["page"] >= 1  # 1-indexed
    assert chunk.metadata["chunk_id"] == "hash999_2"
    assert chunk.metadata["doc_hash"] == "hash999"
    assert chunk.chunk_id == "hash999_2"


def test_retriever_controlled_empty_collection(controlled_chroma):
    """Valida que coleção vazia retorna lista vazia sem lançar exceções."""
    retriever = VectorRetriever(vector_store=controlled_chroma)
    results = retriever.search("query em base vazia")
    assert results == []
    assert isinstance(results, list)


def test_retriever_score_is_raw_distance_not_normalized_relevance(controlled_chroma):
    """Comprova que o score retornado é a distância bruta cosseno (0 para idêntico, ~1 para ortogonal), não relevância normalizada."""
    docs = [
        Document(
            page_content="Texto identico exato.",
            metadata={"file_name": "id.pdf", "page": 1, "chunk_id": "c_id", "source": "id.pdf", "doc_hash": "h_id"}
        ),
        Document(
            page_content="Texto ortogonal exato.",
            metadata={"file_name": "orto.pdf", "page": 1, "chunk_id": "c_orto", "source": "orto.pdf", "doc_hash": "h_orto"}
        ),
    ]
    controlled_chroma.add_documents(docs)

    retriever = VectorRetriever(vector_store=controlled_chroma, settings=Settings(retrieval_k=2, relevance_threshold=1.0))
    results = retriever.search("query de teste")

    assert len(results) == 2
    chunk_identico = next(r for r in results if "identico" in r.content)
    chunk_ortogonal = next(r for r in results if "ortogonal" in r.content)

    # Distância cosseno: vetor idêntico tem distância 0.0
    assert chunk_identico.score == pytest.approx(0.0, abs=1e-4)
    # Distância cosseno: vetor ortogonal tem distância 1.0
    assert chunk_ortogonal.score == pytest.approx(1.0, abs=1e-4)

    # Menor score indica maior proximidade (distância menor é melhor)
    assert chunk_identico.score < chunk_ortogonal.score
