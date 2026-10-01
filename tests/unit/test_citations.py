import pytest
from src.generation.citations import CitationFormatter, Citation
from src.retrieval.search import RetrievedChunk


def test_format_citations_list():
    """Valida formatação estruturada de fontes consultadas."""
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            content="Python é uma linguagem de alto nível com tipagem dinâmica.",
            metadata={"file_name": "python_faq.pdf", "page": 3},
            score=0.88
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            content="Listas em Python são mutáveis e indexadas a partir de zero.",
            metadata={"file_name": "python_faq.pdf", "page": 5},
            score=0.82
        )
    ]

    formatter = CitationFormatter()
    citations = formatter.build_citations(chunks)

    assert len(citations) == 2
    assert citations[0].marker == "[1]"
    assert citations[0].file_name == "python_faq.pdf"
    assert citations[0].page == 3
    assert citations[1].marker == "[2]"

    formatted_text = formatter.format_sources_block(citations)
    assert "[1] python_faq.pdf (pág. 3)" in formatted_text
    assert "[2] python_faq.pdf (pág. 5)" in formatted_text


def test_validate_citations_in_response():
    """Valida que marcadores gerados pelo modelo correspondem a fontes reais recuperadas."""
    formatter = CitationFormatter()
    valid_markers = ["[1]", "[2]"]

    # Caso válido
    valid_text = "Python é dinâmico [1] e possui listas indexadas em zero [2]."
    cleaned_valid, is_ok = formatter.validate_response_citations(valid_text, valid_markers)
    assert is_ok is True
    assert "[1]" in cleaned_valid
    assert "[2]" in cleaned_valid

    # Caso com alucinação de marcador inexistente [99]
    hallucinated_text = "Esta afirmação cita uma fonte fantasma [99]."
    cleaned_invalid, is_ok = formatter.validate_response_citations(hallucinated_text, valid_markers)
    assert is_ok is False
    assert "[99]" not in cleaned_invalid  # Marcador fantasma é removido/sanitizado
