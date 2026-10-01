import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from src.core.exceptions import ConfigurationError

# Carrega variáveis do arquivo .env caso exista
load_dotenv()


class Settings(BaseModel):
    """Configurações centralizadas do HSS AI Assistant."""

    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    documents_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENTS_DIRECTORY", "base")
    )

    chroma_persist_directory: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", "db")
    )
    retrieval_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4"))
    )
    relevance_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
    )

    langchain_tracing_v2: bool = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
        in ("true", "1", "yes")
    )
    langchain_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", None)
    )
    langchain_project: str = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "hss-ai-assistant")
    )
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigurationError("chunk_size deve ser maior que zero.")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ConfigurationError("chunk_overlap não pode ser negativo.")
        return v

    @field_validator("relevance_threshold")
    @classmethod
    def validate_relevance_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ConfigurationError("relevance_threshold deve estar entre 0.0 e 1.0.")
        return v

    @field_validator("retrieval_k")
    @classmethod
    def validate_retrieval_k(cls, v: int) -> int:
        if v < 1:
            raise ConfigurationError("retrieval_k deve ser pelo menos 1.")
        return v

    @model_validator(mode="after")
    def validate_overlap_vs_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap deve ser menor que chunk_size.")
        return self


@lru_cache()
def get_settings() -> Settings:
    """Retorna uma instância singleton em cache das configurações."""
    return Settings()
