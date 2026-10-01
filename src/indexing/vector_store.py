import logging
from typing import Dict, List, Optional, Set
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from src.core.config import Settings, get_settings
from src.core.exceptions import VectorStoreError
from src.indexing.loader import PDFLoader
from src.indexing.splitter import DocumentSplitter

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Gerenciador de sincronização idempotente com ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        embeddings: Optional[Embeddings] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = Chroma(
                persist_directory=self.settings.chroma_persist_directory,
                embedding_function=self.embeddings,
            )

        self.loader = PDFLoader()
        self.splitter = DocumentSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def get_total_vectors(self) -> int:
        """Retorna o número total de vetores persistidos na coleção."""
        try:
            return self.vector_store._collection.count()
        except Exception as e:
            logger.error(f"Erro ao contar vetores na coleção: {e}")
            raise VectorStoreError(f"Falha ao acessar coleção do ChromaDB: {e}") from e

    def _get_existing_metadata(self) -> List[dict]:
        """Obtém os metadados de todos os itens cadastrados no ChromaDB."""
        try:
            results = self.vector_store.get(include=["metadatas"])
            return results.get("metadatas", []) or []
        except Exception as e:
            logger.error(f"Erro ao obter metadados existentes do ChromaDB: {e}")
            return []

    def sync_directory(self, directory: Optional[str] = None) -> Dict[str, int]:
        """Sincroniza documentos de um diretório com o ChromaDB de forma idempotente.

        - Ignora arquivos inalterados (0 chamadas a embeddings).
        - Atualiza arquivos modificados.
        - Remove vetores órfãos de arquivos deletados da pasta.

        Returns:
            Dicionário com o resumo da sincronização {'added', 'updated', 'removed', 'skipped'}.
        """
        dir_to_scan = directory or self.settings.documents_directory
        documents = self.loader.load_documents(dir_to_scan)

        # 1. Identificar arquivos presentes no diretório atual
        current_files: Set[str] = {
            doc.metadata.get("file_name", "") for doc in documents if "file_name" in doc.metadata
        }

        # 2. Obter estado atual no ChromaDB
        existing_metas = self._get_existing_metadata()
        existing_files: Set[str] = {
            m.get("file_name", "") for m in existing_metas if "file_name" in m
        }
        existing_chunk_ids: Set[str] = {
            m.get("chunk_id", "") for m in existing_metas if "chunk_id" in m
        }

        summary = {"added": 0, "updated": 0, "removed": 0, "skipped": 0}

        # 3. Remoção de órfãos (arquivos no Chroma que não estão mais no diretório)
        orphan_files = existing_files - current_files
        for orphan in orphan_files:
            if not orphan:
                continue
            logger.info(f"Removendo vetores órfãos do arquivo excluído: {orphan}")
            try:
                self.vector_store.delete(where={"file_name": orphan})
                summary["removed"] += 1
            except Exception as e:
                logger.warning(f"Não foi possível remover órfãos para {orphan}: {e}")

        if not documents:
            return summary

        # 4. Agrupar documentos por arquivo para verificar idempotência
        docs_by_file: Dict[str, List[Document]] = {}
        for doc in documents:
            fname = doc.metadata.get("file_name", "")
            docs_by_file.setdefault(fname, []).append(doc)

        for fname, file_docs in docs_by_file.items():
            chunks = self.splitter.split_documents(file_docs)
            if not chunks:
                continue

            new_chunk_ids = [c.metadata.get("chunk_id", "") for c in chunks]

            # Verificar se todos os chunks já existem identicamente no banco
            all_chunks_exist = all(cid in existing_chunk_ids for cid in new_chunk_ids)

            if all_chunks_exist and fname in existing_files:
                # Arquivo inalterado: idempotência total
                logger.debug(f"Arquivo inalterado ignorado: {fname}")
                summary["skipped"] += len(chunks)
                continue

            # Se o arquivo já existia mas com hashes diferentes -> Atualização
            if fname in existing_files:
                logger.info(f"Atualizando documento modificado: {fname}")
                try:
                    self.vector_store.delete(where={"file_name": fname})
                    summary["updated"] += 1
                except Exception as e:
                    logger.warning(f"Erro ao limpar versão antiga de {fname}: {e}")

            # Inserir novos chunks
            logger.info(f"Indexando {len(chunks)} chunks para o arquivo: {fname}")
            self.vector_store.add_documents(chunks, ids=new_chunk_ids)
            summary["added"] += len(chunks)

        logger.info(f"Sincronização concluída: {summary}")
        return summary
