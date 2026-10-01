import pytest
from unittest.mock import MagicMock
from src.generation.recontextualizer import QueryRecontextualizer


def test_recontextualizer_empty_history():
    """Valida que pergunta sem histórico anterior não aciona chamada de LLM."""
    mock_llm = MagicMock()
    recontextualizer = QueryRecontextualizer(llm=mock_llm)

    result = recontextualizer.recontextualize("O que é Python?", history=[])

    assert result == "O que é Python?"
    mock_llm.invoke.assert_not_called()


def test_recontextualizer_with_history():
    """Valida reformulação da pergunta quando há histórico anterior."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Qual é a duração do curso de Python?")

    recontextualizer = QueryRecontextualizer(llm=mock_llm)
    history = [
        {"user": "Gostaria de saber sobre o curso de Python.", "assistant": "O curso aborda sintaxe e POO."}
    ]

    result = recontextualizer.recontextualize("Qual é a duração dele?", history=history)

    assert result == "Qual é a duração do curso de Python?"
    mock_llm.invoke.assert_called_once()
