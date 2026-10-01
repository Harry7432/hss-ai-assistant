import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.indexing.loader import PDFLoader
from src.core.exceptions import IngestionError


def test_load_directory_nonexistent():
    """Valida que diretório inexistente levanta IngestionError."""
    loader = PDFLoader()
    with pytest.raises(IngestionError, match="Diretório de documentos não encontrado"):
        loader.load_documents("diretorio_que_nao_existe_12345")


def test_load_documents_success(tmp_path):
    """Valida carregamento de documentos e padronização de metadados."""
    # Criar um PDF simulado
    pdf_file = tmp_path / "documento_teste.pdf"
    pdf_file.write_text("conteudo simulado")

    fake_doc = Document(
        page_content="Texto da página 1 do FAQ de Python",
        metadata={"source": str(pdf_file), "page": 0}
    )

    with patch("src.indexing.loader.PyPDFDirectoryLoader") as mock_dir_loader:
        instance = mock_dir_loader.return_value
        instance.load.return_value = [fake_doc]

        loader = PDFLoader()
        docs = loader.load_documents(str(tmp_path))

        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "Texto da página 1 do FAQ de Python"
        assert doc.metadata["file_name"] == "documento_teste.pdf"
        assert doc.metadata["page"] == 1  # 1-indexed padronizado
        assert "source" in doc.metadata


def test_load_documents_empty_directory(tmp_path):
    """Valida comportamento quando diretório existe mas não contém PDFs."""
    with patch("src.indexing.loader.PyPDFDirectoryLoader") as mock_dir_loader:
        instance = mock_dir_loader.return_value
        instance.load.return_value = []

        loader = PDFLoader()
        docs = loader.load_documents(str(tmp_path))
        assert docs == []
