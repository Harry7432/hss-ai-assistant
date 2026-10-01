import os
import pytest
from src.core.config import Settings, get_settings
from src.core.exceptions import ConfigurationError


def test_settings_default_values(monkeypatch):
    """Test default values of Settings when no env vars are provided."""
    # Clear environment variables for testing defaults
    for key in [
        "OPENAI_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL",
        "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K",
        "RELEVANCE_THRESHOLD", "DOCUMENTS_DIRECTORY", "CHROMA_PERSIST_DIRECTORY"
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings()
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 4
    assert settings.relevance_threshold == 0.7
    assert settings.documents_directory == "base"
    assert settings.chroma_persist_directory == "db"


def test_settings_custom_values(monkeypatch):
    """Test loading custom configuration values from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RETRIEVAL_K", "8")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.85")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    settings = Settings()
    assert settings.openai_api_key == "test-key-123"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100
    assert settings.retrieval_k == 8
    assert settings.relevance_threshold == 0.85
    assert settings.llm_model == "gpt-4o"


def test_settings_validation_invalid_chunk_overlap():
    """Test that chunk_overlap >= chunk_size raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="chunk_overlap deve ser menor que chunk_size"):
        Settings(chunk_size=500, chunk_overlap=500)


def test_settings_validation_invalid_threshold():
    """Test that invalid relevance threshold raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="relevance_threshold deve estar entre 0.0 e 1.0"):
        Settings(relevance_threshold=1.5)


def test_get_settings_singleton():
    """Test that get_settings() returns a cached Settings instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
