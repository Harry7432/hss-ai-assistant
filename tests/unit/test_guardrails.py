import pytest
import re
from unittest.mock import MagicMock, patch
from src.generation.rag_engine import RAGEngine
from src.observability.logging import sanitize_log_message
from src.core.exceptions import GenerationError


def test_sanitize_log_message_masks_api_keys():
    """Valida que chaves no formato sk-... são mascaradas nos logs."""
    raw_message = "Enviando requisição com key sk-abcdef1234567890uvwxyz para a OpenAI."
    sanitized = sanitize_log_message(raw_message)

    assert "sk-abcdef1234567890uvwxyz" not in sanitized
    assert "sk-***" in sanitized


def test_rag_engine_fallback_when_no_chunks(in_memory_chroma, mock_llm_factory):
    """Valida acionamento do fallback estrito quando não há chunks relevantes."""
    llm = mock_llm_factory(["Resposta que não deveria ser chamada"])
    engine = RAGEngine(vector_store=in_memory_chroma, llm=llm)

    # Forçar busca que retorna vazio
    with patch.object(engine.retriever, "search", return_value=[]):
        response = engine.query("Qual o segredo do universo?")

        assert response.is_fallback is True
        assert "Não foi possível encontrar informações suficientes" in response.answer
        assert len(response.citations) == 0


def test_rag_engine_retries_on_transient_error(in_memory_chroma):
    """Valida política de retries contra falhas transitórias de conexão com a OpenAI."""
    mock_llm = MagicMock()
    # Simular falha nas 2 primeiras chamadas e sucesso na 3ª
    mock_llm.invoke.side_effect = [
        Exception("Rate limit 429"),
        Exception("Connection timeout"),
        MagicMock(content="Resposta gerada após retry [1].")
    ]

    engine = RAGEngine(vector_store=in_memory_chroma, llm=mock_llm)

    chunk = MagicMock(content="Conteúdo válido", metadata={"file_name": "doc.pdf", "page": 1}, score=0.9)
    with patch.object(engine.retriever, "search", return_value=[chunk]):
        response = engine.query("Pergunta com retry")
        assert response.is_fallback is False
        assert mock_llm.invoke.call_count == 3
