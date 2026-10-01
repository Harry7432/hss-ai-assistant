import logging
import pytest
from pathlib import Path
from unittest.mock import patch
from langchain_core.documents import Document
from src.indexing.loader import PDFLoader


def test_load_directory_nonexistent(caplog):
    """Valida que diretório inexistente emite warning e retorna 0 documentos sem levantar exceção."""
    loader = PDFLoader()
    with caplog.at_level(logging.WARNING):
        docs = loader.load_documents("diretorio_que_nao_existe_12345")
    assert docs == []
    assert any("não encontrado" in record.message or "ausente" in record.message for record in caplog.records)


def test_load_documents_empty_directory(tmp_path, caplog):
    """Valida que diretório vazio emite warning e retorna 0 documentos sem erro."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    loader = PDFLoader()
    with caplog.at_level(logging.WARNING):
        docs = loader.load_documents(str(empty_dir))
    assert docs == []
    assert any("vazio" in record.message or "nenhum" in record.message.lower() for record in caplog.records)


def test_load_documents_success(tmp_path):
    """Valida carregamento de documentos e padronização de metadados com página 1-indexed."""
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
        assert doc.metadata["page"] == 1  # 1-indexed: primeira página física é 1
        assert doc.metadata["source"] == str(pdf_file)


def test_load_documents_multi_page_1_indexed(tmp_path):
    """Valida que páginas 0-indexed do PyPDF são convertidas para 1-indexed sequencialmente."""
    pdf_file = tmp_path / "apostila.pdf"
    pdf_file.write_text("conteudo")

    fake_docs = [
        Document(page_content="Página 1", metadata={"source": str(pdf_file), "page": 0}),
        Document(page_content="Página 2", metadata={"source": str(pdf_file), "page": 1}),
        Document(page_content="Página 3", metadata={"source": str(pdf_file), "page": 2}),
    ]

    with patch("src.indexing.loader.PyPDFDirectoryLoader") as mock_dir_loader:
        instance = mock_dir_loader.return_value
        instance.load.return_value = fake_docs

        loader = PDFLoader()
        docs = loader.load_documents(str(tmp_path))

        assert len(docs) == 3
        assert [d.metadata["page"] for d in docs] == [1, 2, 3]


def test_load_documents_pdf_without_text_ignored(tmp_path, caplog):
    """Valida que PDFs escaneados ou sem texto extraível emitem warning e são ignorados."""
    pdf_empty = tmp_path / "escaneado.pdf"
    pdf_empty.write_text("fake pdf")
    pdf_valid = tmp_path / "valido.pdf"
    pdf_valid.write_text("fake pdf valido")

    fake_docs = [
        Document(page_content="", metadata={"source": str(pdf_empty), "page": 0}),
        Document(page_content="   \n\t  ", metadata={"source": str(pdf_empty), "page": 1}),
        Document(page_content="Texto extraível legítimo", metadata={"source": str(pdf_valid), "page": 0}),
    ]

    with patch("src.indexing.loader.PyPDFDirectoryLoader") as mock_dir_loader:
        instance = mock_dir_loader.return_value
        instance.load.return_value = fake_docs

        loader = PDFLoader()
        with caplog.at_level(logging.WARNING):
            docs = loader.load_documents(str(tmp_path))

        # Apenas o documento com texto legítimo deve ser retornado
        assert len(docs) == 1
        assert docs[0].page_content == "Texto extraível legítimo"
        assert docs[0].metadata["file_name"] == "valido.pdf"
        assert docs[0].metadata["page"] == 1

        # Deve ter logado warning para as páginas sem texto
        assert any("sem texto" in record.message or "ignorado" in record.message.lower() for record in caplog.records)
