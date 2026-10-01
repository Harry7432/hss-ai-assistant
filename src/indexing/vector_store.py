from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.core.config import Settings, get_settings
from src.core.exceptions import VectorStoreError
from src.indexing.loader import LoadResult, PDFLoader
from src.indexing.manifest import (
    IndexManifest,
    build_fingerprint,
    check_prune_safety,
    create_sync_plan,
    reconcile_manifest_with_chroma,
)
from src.indexing.splitter import DocumentSplitter, compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Relatório estruturado com contagens e métricas da sincronização."""
    indexed: int = 0
    unchanged: int = 0
    renamed: int = 0
    ignored_copies: int = 0
    replaced: int = 0
    pruned: int = 0
    protected: int = 0
    prune_refused: int = 0
    embedding_calls: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    def __getitem__(self, item: str) -> int:
        return getattr(self, item)


class CountingEmbeddings(Embeddings):
    """Proxy para rastreamento determinístico de chamadas à API de embeddings."""

    def __init__(self, inner: Embeddings):
        self._inner = inner
        self.call_count = getattr(inner, "call_count", 0)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self.call_count += 1
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        self.call_count += 1
        return self._inner.embed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class VectorStoreManager:
    """Gerenciador de sincronização idempotente com ChromaDB e manifesto de integridade."""

    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        embeddings: Optional[Embeddings] = None,
        settings: Optional[Settings] = None,
        client: Optional[chromadb.ClientAPI] = None,
        manifest_path: Optional[Path | str] = None,
    ):
        self.settings = settings or get_settings()

        if embeddings is not None:
            if hasattr(embeddings, "call_count"):
                self.embeddings = embeddings
            else:
                self.embeddings = CountingEmbeddings(embeddings)
        elif vector_store is not None and getattr(vector_store, "_embedding_function", None) is not None:
            emb_func = getattr(vector_store, "_embedding_function")
            if hasattr(emb_func, "call_count"):
                self.embeddings = emb_func
            else:
                self.embeddings = CountingEmbeddings(emb_func)
        else:
            real_emb = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                openai_api_key=self.settings.openai_api_key,
            )
            self.embeddings = CountingEmbeddings(real_emb)

        self.collection_name = self.settings.collection_name

        if vector_store is not None:
            self.vector_store = vector_store
            self.client = getattr(vector_store, "_client", None)
            if hasattr(vector_store, "_collection") and vector_store._collection is not None:
                self.collection_name = vector_store._collection.name
        else:
            if client is not None:
                self.client = client
            else:
                self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_directory)

            try:
                self.client.get_or_create_collection(
                    name=self.collection_name,
                    configuration={"hnsw": {"space": "cosine"}},
                )
            except Exception as e:
                logger.debug(f"Coleção já existente ou obtida: {e}")

            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )

        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else Path(self.settings.chroma_persist_directory) / "manifest.json"
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

    def get_all_chunk_ids(self) -> Set[str]:
        """Obtém todos os IDs registrados na coleção atual do ChromaDB."""
        try:
            res = self.vector_store.get(include=[])
            return set(res.get("ids", []) or [])
        except Exception as e:
            logger.warning(f"Erro ao obter IDs da coleção ChromaDB: {e}")
            return set()

    def _get_existing_metadata(self) -> List[dict]:
        """Obtém os metadados de todos os itens cadastrados no ChromaDB."""
        try:
            results = self.vector_store.get(include=["metadatas"])
            return results.get("metadatas", []) or []
        except Exception as e:
            logger.error(f"Erro ao obter metadados existentes do ChromaDB: {e}")
            return []

    def _reset_collection(self) -> None:
        """Apaga e recria a coleção no Chroma com a métrica cosseno."""
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.debug(f"Coleção não existia para ser deletada: {e}")
            self.client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            all_ids = self.get_all_chunk_ids()
            if all_ids:
                self.vector_store.delete(ids=list(all_ids))

    def _full_reindex(
        self,
        docs_by_file: Dict[str, List[Document]],
        files_on_disk: Dict[str, str],
        current_fingerprint: Dict[str, Any],
        start_calls: int,
    ) -> SyncReport:
        """Executa a reindexação total do acervo, gravando o manifesto por último."""
        self._reset_collection()
        new_manifest = IndexManifest(fingerprint=current_fingerprint)
        report = SyncReport()

        for file_path, file_docs in docs_by_file.items():
            doc_hash = files_on_disk[file_path]
            if doc_hash in new_manifest.files:
                logger.warning(f"Cópia duplicada ignorada na reindexação: {file_path}")
                report.ignored_copies += 1
                continue

            chunks = self.splitter.split_documents(file_docs)
            if not chunks:
                continue

            chunk_ids = [c.metadata.get("chunk_id", "") for c in chunks]
            self.vector_store.add_documents(chunks, ids=chunk_ids)

            file_name = (
                file_docs[0].metadata.get("file_name", Path(file_path).name)
                if file_docs
                else Path(file_path).name
            )
            new_manifest.files[doc_hash] = {
                "paths": [file_path],
                "file_name": file_name,
                "chunk_ids": chunk_ids,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            report.indexed += 1

        new_manifest.save_atomic(self.manifest_path)
        end_calls = getattr(self.embeddings, "call_count", 0)
        report.embedding_calls = end_calls - start_calls
        return report

    def sync_directory(
        self,
        *,
        prune: bool = False,
    ) -> SyncReport:
        """Sincroniza documentos de um diretório com o ChromaDB via manifesto e integridade SHA-256.

        Ordem da sincronização:
        1. Ler diretório com loader (LoadResult) e calcular doc_hash dos arquivos lidos.
        2. Carregar manifesto. Fingerprint inválido, ausente ou divergência com Chroma = REINDEXAÇÃO TOTAL.
        3. Caso contrário, montar SyncPlan e aplicar:
           - unchanged: zero chamadas de embedding.
           - renamed: atualizar metadados no Chroma e caminhos no manifesto, zero chamadas.
           - ignored_copies: warning e ignorar.
           - to_index e replaced: inserir chunks novos ANTES de apagar os antigos.
           - orphans: usar check_prune_safety (recusado = registra prune_refused e segue).
           - protected: nunca apagar.
        4. Gravar manifesto POR ÚLTIMO, de forma atômica.
        """
        dir_to_scan = self.settings.documents_directory
        start_calls = getattr(self.embeddings, "call_count", 0)

        # 1. Ler o diretório com o loader (LoadResult) e calcular o doc_hash de cada arquivo lido
        load_result: LoadResult = self.loader.load_documents(dir_to_scan)
        failed_files: Set[str] = set(load_result.failed_files)

        docs_by_file: Dict[str, List[Document]] = {}
        for doc in load_result.documents:
            source = doc.metadata.get("source", "")
            if source:
                docs_by_file.setdefault(source, []).append(doc)

        files_on_disk: Dict[str, str] = {}
        for file_path, file_docs in docs_by_file.items():
            fallback = file_docs[0].page_content if file_docs else ""
            files_on_disk[file_path] = compute_file_hash(file_path, fallback_content=fallback)

        # 2. Carregar o manifesto e verificar fingerprint e reconciliação
        manifest = IndexManifest.load(self.manifest_path)
        current_fingerprint = build_fingerprint(self.settings)

        chroma_ids = self.get_all_chunk_ids()
        is_fp_valid = manifest.is_fingerprint_valid(current_fingerprint)
        is_reconciled = reconcile_manifest_with_chroma(manifest, chroma_ids) if is_fp_valid else False

        # Fingerprint inválido, manifesto ausente ou divergência com os IDs do Chroma = REINDEXAÇÃO TOTAL
        if not is_fp_valid or not is_reconciled:
            logger.info(
                "Disparando REINDEXAÇÃO TOTAL (fingerprint inválido, manifesto ausente ou divergência com Chroma)."
            )
            return self._full_reindex(
                docs_by_file=docs_by_file,
                files_on_disk=files_on_disk,
                current_fingerprint=current_fingerprint,
                start_calls=start_calls,
            )

        # 3. Caso contrário, montar o SyncPlan e aplicar
        plan = create_sync_plan(
            files_on_disk=files_on_disk,
            failed_files=failed_files,
            manifest=manifest,
        )

        report = SyncReport(
            unchanged=len(plan.unchanged),
            ignored_copies=len(plan.ignored_copies),
            protected=len(plan.protected),
        )

        # - renamed: atualizar file_name e source nos metadados do Chroma e os paths no manifesto (0 embeddings)
        for rename in plan.renamed:
            chunk_ids = manifest.files.get(rename.doc_hash, {}).get("chunk_ids", [])
            if chunk_ids:
                try:
                    c_data = self.vector_store._collection.get(ids=chunk_ids, include=["metadatas"])
                    old_metas = c_data.get("metadatas", []) or []
                    updated_metas = []
                    for m in old_metas:
                        new_m = dict(m)
                        new_m["file_name"] = rename.file_name
                        new_m["source"] = rename.new_path
                        updated_metas.append(new_m)
                    self.vector_store._collection.update(ids=chunk_ids, metadatas=updated_metas)
                except Exception as e:
                    logger.warning(f"Erro ao atualizar metadados de arquivo renomeado {rename.file_name}: {e}")

            if rename.doc_hash in manifest.files:
                manifest.files[rename.doc_hash]["paths"] = [rename.new_path]
                manifest.files[rename.doc_hash]["file_name"] = rename.file_name
                manifest.files[rename.doc_hash]["source"] = rename.new_path
            report.renamed += 1

        # Mapeamento auxiliar de caminho para doc_hash no manifesto
        manifest_path_to_hash: Dict[str, str] = {}
        for doc_hash, entry in manifest.files.items():
            for p in entry.get("paths", []):
                manifest_path_to_hash[p] = doc_hash

        # - to_index e replaced: inserir os chunks novos ANTES de apagar os antigos
        for file_path, doc_hash in plan.to_index.items():
            file_docs = docs_by_file.get(file_path, [])
            chunks = self.splitter.split_documents(file_docs)
            if not chunks:
                continue

            new_chunk_ids = [c.metadata.get("chunk_id", "") for c in chunks]

            # Inserir chunks novos ANTES de apagar os antigos
            self.vector_store.add_documents(chunks, ids=new_chunk_ids)

            # Se for replaced, apagar os chunks da versão anterior
            old_hash = manifest_path_to_hash.get(file_path)
            if old_hash and old_hash in plan.replaced:
                old_chunk_ids = manifest.files.get(old_hash, {}).get("chunk_ids", [])
                if old_chunk_ids:
                    try:
                        self.vector_store.delete(ids=old_chunk_ids)
                    except Exception as e:
                        logger.warning(f"Erro ao deletar versão antiga de chunks {old_hash}: {e}")
                if old_hash in manifest.files:
                    del manifest.files[old_hash]
                report.replaced += 1
            else:
                report.indexed += 1

            file_name = (
                file_docs[0].metadata.get("file_name", Path(file_path).name)
                if file_docs
                else Path(file_path).name
            )
            manifest.files[doc_hash] = {
                "paths": [file_path],
                "file_name": file_name,
                "chunk_ids": new_chunk_ids,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }

        # - orphans: usar check_prune_safety. Recusado = não apaga, registra prune_refused e segue
        if plan.orphans:
            safe, msg = check_prune_safety(
                plan=plan,
                max_prune_percentage=self.settings.max_prune_percentage,
                prune=prune,
            )
            if safe:
                for orphan_hash in plan.orphans:
                    orphan_chunk_ids = manifest.files.get(orphan_hash, {}).get("chunk_ids", [])
                    if orphan_chunk_ids:
                        try:
                            self.vector_store.delete(ids=orphan_chunk_ids)
                        except Exception as e:
                            logger.warning(f"Erro ao podar chunks do órfão {orphan_hash}: {e}")
                    if orphan_hash in manifest.files:
                        del manifest.files[orphan_hash]
                    report.pruned += 1
            else:
                logger.warning(f"Poda de órfãos recusada pela trava de segurança: {msg}")
                report.prune_refused = len(plan.orphans)

        # 4. Gravar o manifesto POR ÚLTIMO, de forma atômica
        manifest.fingerprint = current_fingerprint
        manifest.save_atomic(self.manifest_path)

        end_calls = getattr(self.embeddings, "call_count", 0)
        report.embedding_calls = end_calls - start_calls
        return report
