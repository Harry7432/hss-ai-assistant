import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.indexing.loader import PDFLoader, LoadResult


def test_load_directory_nonexistent(caplog):
    """Valida que diretório inexistente emite warning e retorna LoadResult vazio sem levantar exceção."""
    loader = PDFLoader()
    with caplog.at_level(logging.WARNING):
        result = loader.load_documents("diretorio_que_nao_existe_12345")
    assert isinstance(result, LoadResult)
    assert result.documents == []
    assert result.failed_files == []
    assert result.skipped_no_text == []
    assert any("não encontrado" in record.message or "ausente" in record.message for record in caplog.records)


def test_load_documents_empty_directory(tmp_path, caplog):
    """Valida que diretório vazio emite warning e retorna LoadResult vazio sem erro."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    loader = PDFLoader()
    with caplog.at_level(logging.WARNING):
        result = loader.load_documents(str(empty_dir))
    assert isinstance(result, LoadResult)
    assert result.documents == []
    assert result.failed_files == []
    assert result.skipped_no_text == []
    assert any("vazio" in record.message or "nenhum" in record.message.lower() for record in caplog.records)


def test_load_documents_success(tmp_path):
    """Valida carregamento de documentos e padronização de metadados com página 1-indexed."""
    pdf_file = tmp_path / "documento_teste.pdf"
    pdf_file.write_text("conteudo simulado")

    fake_doc = Document(
        page_content="Texto da página 1 do FAQ de Python",
        metadata={"source": str(pdf_file), "page": 0}
    )

    with patch("src.indexing.loader.PyPDFLoader") as mock_pdf_loader:
        instance = mock_pdf_loader.return_value
        instance.load.return_value = [fake_doc]

        loader = PDFLoader()
        result = loader.load_documents(str(tmp_path))

        assert isinstance(result, LoadResult)
        assert len(result.documents) == 1
        doc = result.documents[0]
        assert doc.page_content == "Texto da página 1 do FAQ de Python"
        assert doc.metadata["file_name"] == "documento_teste.pdf"
        assert doc.metadata["page"] == 1  # 1-indexed: primeira página física é 1
        assert doc.metadata["source"] == str(pdf_file)
        assert result.failed_files == []
        assert result.skipped_no_text == []


def test_load_documents_multi_page_1_indexed(tmp_path):
    """Valida que páginas 0-indexed do PyPDF são convertidas para 1-indexed sequencialmente."""
    pdf_file = tmp_path / "apostila.pdf"
    pdf_file.write_text("conteudo")

    fake_docs = [
        Document(page_content="Página 1", metadata={"source": str(pdf_file), "page": 0}),
        Document(page_content="Página 2", metadata={"source": str(pdf_file), "page": 1}),
        Document(page_content="Página 3", metadata={"source": str(pdf_file), "page": 2}),
    ]

    with patch("src.indexing.loader.PyPDFLoader") as mock_pdf_loader:
        instance = mock_pdf_loader.return_value
        instance.load.return_value = fake_docs

        loader = PDFLoader()
        result = loader.load_documents(str(tmp_path))

        assert isinstance(result, LoadResult)
        assert len(result.documents) == 3
        assert [d.metadata["page"] for d in result.documents] == [1, 2, 3]
        assert result.failed_files == []
        assert result.skipped_no_text == []


def test_load_documents_pdf_without_text_ignored(tmp_path, caplog):
    """Valida que PDFs escaneados ou sem texto extraível entram em skipped_no_text e emitem warning."""
    pdf_empty = tmp_path / "escaneado.pdf"
    pdf_empty.write_text("fake pdf")
    pdf_valid = tmp_path / "valido.pdf"
    pdf_valid.write_text("fake pdf valido")

    def side_effect(file_path):
        mock_instance = MagicMock()
        if "escaneado" in file_path:
            mock_instance.load.return_value = [
                Document(page_content="", metadata={"source": str(pdf_empty), "page": 0}),
                Document(page_content="   \n\t  ", metadata={"source": str(pdf_empty), "page": 1}),
            ]
        else:
            mock_instance.load.return_value = [
                Document(page_content="Texto extraível legítimo", metadata={"source": str(pdf_valid), "page": 0})
            ]
        return mock_instance

    with patch("src.indexing.loader.PyPDFLoader", side_effect=side_effect):
        loader = PDFLoader()
        with caplog.at_level(logging.WARNING):
            result = loader.load_documents(str(tmp_path))

        assert isinstance(result, LoadResult)
        assert len(result.documents) == 1
        assert result.documents[0].page_content == "Texto extraível legítimo"
        assert result.documents[0].metadata["file_name"] == "valido.pdf"
        assert result.documents[0].metadata["page"] == 1
        assert str(pdf_empty) in result.skipped_no_text
        assert result.failed_files == []
        assert any("sem texto" in record.message or "ignorada" in record.message.lower() for record in caplog.records)


def test_load_documents_corrupted_pdf_recorded_in_failed_files_and_continues(tmp_path, caplog):
    """Valida que PDF corrompido gera warning, NÃO levanta exceção, entra em failed_files e não interrompe os demais."""
    pdf_corrupt = tmp_path / "corrompido.pdf"
    pdf_corrupt.write_text("conteudo corrompido")
    pdf_valid = tmp_path / "valido.pdf"
    pdf_valid.write_text("conteudo valido")

    def side_effect(file_path):
        mock_instance = MagicMock()
        if "corrompido" in file_path:
            mock_instance.load.side_effect = Exception("PDF stream is corrupted or unreadable")
        else:
            mock_instance.load.return_value = [
                Document(page_content="Texto legítimo do PDF válido", metadata={"source": str(pdf_valid), "page": 0})
            ]
        return mock_instance

    with patch("src.indexing.loader.PyPDFLoader", side_effect=side_effect):
        loader = PDFLoader()
        with caplog.at_level(logging.WARNING):
            result = loader.load_documents(str(tmp_path))

        assert isinstance(result, LoadResult)
        assert str(pdf_corrupt) in result.failed_files
        assert result.skipped_no_text == []
        assert len(result.documents) == 1
        assert result.documents[0].page_content == "Texto legítimo do PDF válido"
        assert result.documents[0].metadata["file_name"] == "valido.pdf"
        assert any("falha ao carregar" in record.message.lower() or "corrompido" in record.message.lower() for record in caplog.records)
