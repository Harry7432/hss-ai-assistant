import pytest
from src.core.exceptions import (
    HSSAssistantError,
    IngestionError,
    VectorStoreError,
    VectorStoreUnavailableError,
    APIConnectionError,
    QuotaExceededError,
)


def test_exception_hierarchy():
    """Valida a hierarquia de exceções de domínio padronizadas."""
    assert issubclass(IngestionError, HSSAssistantError)
    assert issubclass(VectorStoreUnavailableError, (VectorStoreError, HSSAssistantError))
    assert issubclass(APIConnectionError, HSSAssistantError)
    assert issubclass(QuotaExceededError, HSSAssistantError)


def test_raise_custom_exceptions():
    """Valida instanciação e captura correta das exceções com mensagens."""
    with pytest.raises(IngestionError, match="Erro ao carregar PDF"):
        raise IngestionError("Erro ao carregar PDF")

    with pytest.raises(VectorStoreUnavailableError, match="ChromaDB fora do ar"):
        raise VectorStoreUnavailableError("ChromaDB fora do ar")

    with pytest.raises(APIConnectionError, match="Falha de conexão com OpenAI"):
        raise APIConnectionError("Falha de conexão com OpenAI")

    with pytest.raises(QuotaExceededError, match="Cota de API esgotada"):
        raise QuotaExceededError("Cota de API esgotada")
