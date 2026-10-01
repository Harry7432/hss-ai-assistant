"""Módulo de ingestão e indexação vetorial."""
from src.indexing.loader import PDFLoader
from src.indexing.splitter import DocumentSplitter
from src.indexing.vector_store import VectorStoreManager

__all__ = ["PDFLoader", "DocumentSplitter", "VectorStoreManager"]
