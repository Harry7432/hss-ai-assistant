import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from src.core.exceptions import IngestionError

logger = logging.getLogger(__name__)


class PDFLoader:
    """Carregador de documentos PDF com normalização e enriquecimento de metadados."""

    def load_documents(self, directory: str) -> List[Document]:
        """Carrega todos os PDFs de um diretório e normaliza seus metadados.

        Args:
            directory: Caminho para a pasta contendo os PDFs.

        Returns:
            Lista de documentos LangChain com metadados padronizados.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Diretório de documentos não encontrado ou ausente: {directory}")
            return []

        pdf_files = list(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"Diretório vazio ou nenhum PDF encontrado em: {directory}")
            return []

        try:
            loader = PyPDFDirectoryLoader(str(dir_path), glob="*.pdf")
            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"Erro ao carregar PDFs do diretório {directory}: {e}")
            raise IngestionError(f"Falha ao carregar PDFs: {e}") from e

        if not raw_docs:
            logger.warning(f"Nenhum documento extraído do diretório {directory}")
            return []

        normalized_docs: List[Document] = []
        for doc in raw_docs:
            source_path = doc.metadata.get("source", "")
            file_name = Path(source_path).name if source_path else "desconhecido.pdf"
            
            # Validação de conteúdo: se for vazio ou apenas espaços, ignora com warning
            if not doc.page_content or not doc.page_content.strip():
                raw_page = doc.metadata.get("page", 0)
                page_for_log = int(raw_page) + 1 if isinstance(raw_page, (int, str)) else 1
                logger.warning(
                    f"PDF sem texto ou página escaneada ignorada: {file_name} "
                    f"(página {page_for_log}, caminho: {source_path})"
                )
                continue

            # PyPDF retorna 'page' 0-indexed; normalizamos para 1-indexed (primeira página física = 1)
            raw_page = doc.metadata.get("page", 0)
            page_number = int(raw_page) + 1 if isinstance(raw_page, (int, str)) else 1

            meta = dict(doc.metadata)
            meta["file_name"] = file_name
            meta["source"] = str(source_path)
            meta["page"] = page_number

            normalized_docs.append(
                Document(page_content=doc.page_content, metadata=meta)
            )

        logger.info(f"Carregados {len(normalized_docs)} fragmentos de página de {directory}")
        return normalized_docs
