import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Resultado do carregamento de PDFs com relatório de falhas e arquivos ignorados."""

    documents: List[Document] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    skipped_no_text: List[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def __bool__(self) -> bool:
        return bool(self.documents)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, list):
            return self.documents == other
        if isinstance(other, LoadResult):
            return (
                self.documents == other.documents
                and self.failed_files == other.failed_files
                and self.skipped_no_text == other.skipped_no_text
            )
        return False


class PDFLoader:
    """Carregador de documentos PDF com normalização e enriquecimento de metadados."""

    def load_documents(self, directory: str) -> LoadResult:
        """Carrega todos os PDFs de um diretório e normaliza seus metadados.

        Args:
            directory: Caminho para a pasta contendo os PDFs.

        Returns:
            LoadResult contendo documents, failed_files e skipped_no_text.
        """
        result = LoadResult()
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Diretório de documentos não encontrado ou ausente: {directory}")
            return result

        pdf_files = sorted(list(dir_path.glob("*.pdf")))
        if not pdf_files:
            logger.warning(f"Diretório vazio ou nenhum PDF encontrado em: {directory}")
            return result

        for pdf_path in pdf_files:
            str_path = str(pdf_path)
            try:
                loader = PyPDFLoader(str_path)
                raw_docs = loader.load()
            except Exception as e:
                logger.warning(f"Falha ao carregar ou ler PDF corrompido {str_path}: {e}")
                result.failed_files.append(str_path)
                continue

            file_has_text = False
            for doc in raw_docs:
                source_path = doc.metadata.get("source", str_path)
                file_name = Path(source_path).name if source_path else pdf_path.name

                # Validação de conteúdo: se for vazio ou apenas espaços, ignora com warning
                if not doc.page_content or not doc.page_content.strip():
                    raw_page = doc.metadata.get("page", 0)
                    page_for_log = int(raw_page) + 1 if isinstance(raw_page, (int, str)) else 1
                    logger.warning(
                        f"PDF sem texto ou página escaneada ignorada: {file_name} "
                        f"(página {page_for_log}, caminho: {source_path})"
                    )
                    continue

                file_has_text = True
                # PyPDF retorna 'page' 0-indexed; normalizamos para 1-indexed (primeira página física = 1)
                raw_page = doc.metadata.get("page", 0)
                page_number = int(raw_page) + 1 if isinstance(raw_page, (int, str)) else 1

                meta = dict(doc.metadata)
                meta["file_name"] = file_name
                meta["source"] = str(source_path)
                meta["page"] = page_number

                result.documents.append(
                    Document(page_content=doc.page_content, metadata=meta)
                )

            if not file_has_text:
                result.skipped_no_text.append(str_path)

        logger.info(
            f"Carregados {len(result.documents)} fragmentos de página de {directory} "
            f"({len(result.failed_files)} falhas, {len(result.skipped_no_text)} sem texto)"
        )
        return result
