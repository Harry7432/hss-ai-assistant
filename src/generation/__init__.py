"""Módulo de geração de respostas e citações."""
from src.generation.citations import Citation, CitationFormatter
from src.generation.prompt_templates import build_rag_prompt
from src.generation.recontextualizer import QueryRecontextualizer
from src.generation.rag_engine import RAGEngine, RAGResponse

__all__ = [
    "Citation",
    "CitationFormatter",
    "build_rag_prompt",
    "QueryRecontextualizer",
    "RAGEngine",
    "RAGResponse",
]
