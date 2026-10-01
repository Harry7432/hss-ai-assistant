import logging
import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from src.core.config import Settings
from src.indexing.loader import LoadResult
from src.indexing.manifest import IndexManifest
from src.indexing.vector_store import SyncReport, VectorStoreManager


class CountingDeterministicFakeEmbedding(DeterministicFakeEmbedding):
    """Embedding determinístico com contagem de chamadas e capacidade de injeção de falha."""
    call_count: int = 0
    fail_on_embed: bool = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.fail_on_embed:
            raise RuntimeError("Falha injetada no modelo de embeddings")
        self.call_count += 1
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.fail_on_embed:
            raise RuntimeError("Falha injetada no modelo de embeddings")
        self.call_count += 1
        return super().embed_query(text)


@pytest.fixture
def counting_embeddings():
    return CountingDeterministicFakeEmbedding(size=128)


@pytest.fixture
def make_manager(tmp_path, counting_embeddings):
    """Fábrica de VectorStoreManager com coleções e manifestos isolados por UUID."""
    def _create(settings=None, collection_name=None, embeddings=None, manifest_path=None):
        emb = embeddings or counting_embeddings
        col_name = collection_name or f"test_col_{uuid.uuid4().hex}"
        s = settings or Settings(
            collection_name=col_name,
            chroma_persist_directory=str(tmp_path / "chroma"),
            documents_directory=str(tmp_path / "base"),
        )
        # Garante que s.collection_name seja col_name
        s.collection_name = col_name
        client = chromadb.EphemeralClient()
        m_path = manifest_path or (tmp_path / f"manifest_{uuid.uuid4().hex}.json")
        return VectorStoreManager(
            embeddings=emb,
            settings=s,
            client=client,
            manifest_path=m_path,
        )
    return _create


def test_sync_directory_initial_indexing(make_manager, counting_embeddings, tmp_path):
    """1. Indexação inicial: novos arquivos são indexados, chamadas de embedding > 0, manifesto gravado."""
    manager = make_manager()

    doc_file = tmp_path / "base" / "doc1.pdf"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_bytes(b"%PDF-1.4 initial content for doc 1")

    doc = Document(
        page_content="Conteúdo relevante da página 1.",
        metadata={"file_name": "doc1.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        report = manager.sync_directory()

        assert isinstance(report, SyncReport)
        assert report.indexed == 1
        assert report.unchanged == 0
        assert report.embedding_calls > 0
        assert manager.get_total_vectors() > 0

        # Manifesto deve ter sido salvo
        assert manager.manifest_path.exists()
        manifest = IndexManifest.load(manager.manifest_path)
        assert len(manifest.files) == 1


def test_sync_directory_second_run_zero_calls(make_manager, counting_embeddings, tmp_path):
    """2. Segunda execução sem alterações: zero chamadas de embedding e unchanged > 0."""
    manager = make_manager()

    doc_file = tmp_path / "base" / "doc1.pdf"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_bytes(b"%PDF-1.4 fixed content")

    doc = Document(
        page_content="Conteúdo imutável.",
        metadata={"file_name": "doc1.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        # Execução 1
        report1 = manager.sync_directory()
        assert report1.indexed == 1
        initial_calls = counting_embeddings.call_count

        # Execução 2
        report2 = manager.sync_directory()
        assert report2.unchanged == 1
        assert report2.indexed == 0
        assert report2.embedding_calls == 0
        assert counting_embeddings.call_count == initial_calls


def test_sync_directory_renamed_file_zero_calls(make_manager, counting_embeddings, tmp_path):
    """3. Renomeação de arquivo: atualiza file_name e source no Chroma e caminhos no manifesto com zero chamadas."""
    manager = make_manager()

    old_file = tmp_path / "base" / "antigo.pdf"
    old_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"%PDF-1.4 same bytes for rename"
    old_file.write_bytes(pdf_bytes)

    doc_old = Document(
        page_content="Texto idêntico antes e depois.",
        metadata={"file_name": "antigo.pdf", "page": 1, "source": str(old_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_old])):
        manager.sync_directory()
    
    calls_after_first = counting_embeddings.call_count

    # Agora o arquivo foi renomeado no disco: antigo.pdf sumiu e novo.pdf apareceu com mesmo conteúdo
    new_file = tmp_path / "base" / "novo.pdf"
    new_file.write_bytes(pdf_bytes)
    old_file.unlink()

    doc_new = Document(
        page_content="Texto idêntico antes e depois.",
        metadata={"file_name": "novo.pdf", "page": 1, "source": str(new_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_new])):
        report = manager.sync_directory()

        assert report.renamed == 1
        assert report.indexed == 0
        assert report.embedding_calls == 0
        assert counting_embeddings.call_count == calls_after_first

        # Metadados no Chroma devem estar atualizados para novo.pdf
        metas = manager.vector_store.get(include=["metadatas"])["metadatas"]
        assert all(m["file_name"] == "novo.pdf" for m in metas)
        assert all(m["source"] == str(new_file) for m in metas)

        # Manifesto deve registrar o novo caminho
        manifest = IndexManifest.load(manager.manifest_path)
        entry = next(iter(manifest.files.values()))
        assert entry["paths"] == [str(new_file)]
        assert entry["file_name"] == "novo.pdf"


def test_sync_directory_ignored_copies(make_manager, counting_embeddings, tmp_path, caplog):
    """4. Cópia duplicada: emite warning estruturado e descarta cópia sem novas chamadas."""
    manager = make_manager()

    f1 = tmp_path / "base" / "original.pdf"
    f2 = tmp_path / "base" / "copia.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"%PDF-1.4 duplicate bytes"
    f1.write_bytes(pdf_bytes)
    f2.write_bytes(pdf_bytes)

    doc1 = Document(
        page_content="Conteúdo duplicado.",
        metadata={"file_name": "original.pdf", "page": 1, "source": str(f1)}
    )
    doc2 = Document(
        page_content="Conteúdo duplicado.",
        metadata={"file_name": "copia.pdf", "page": 1, "source": str(f2)}
    )

    # Ingestão de original
    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1])):
        manager.sync_directory()

    calls_after_orig = counting_embeddings.call_count

    # Agora sincroniza com original E cópia presentes
    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1, doc2])):
        with caplog.at_level(logging.WARNING):
            report = manager.sync_directory()

        assert report.ignored_copies == 1
        assert report.embedding_calls == 0
        assert counting_embeddings.call_count == calls_after_orig
        assert any("cópia duplicada" in r.message.lower() or "copia" in r.message.lower() for r in caplog.records)


def test_sync_directory_modified_file_insert_before_delete(make_manager, counting_embeddings, tmp_path):
    """5. Arquivo modificado: chunks novos são inseridos ANTES de apagar os antigos."""
    manager = make_manager()

    doc_file = tmp_path / "base" / "arquivo.pdf"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_bytes(b"%PDF-1.4 version 1")

    doc_v1 = Document(
        page_content="Versao 1 inicial.",
        metadata={"file_name": "arquivo.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_v1])):
        manager.sync_directory()

    # Modifica o arquivo no disco
    doc_file.write_bytes(b"%PDF-1.4 version 2 modified")
    doc_v2 = Document(
        page_content="Versao 2 modificada com mais conteudo.",
        metadata={"file_name": "arquivo.pdf", "page": 1, "source": str(doc_file)}
    )

    # Monitorar a ordem de operações: add_documents deve acontecer antes de delete
    op_order = []
    orig_add = manager.vector_store.add_documents
    orig_del = manager.vector_store.delete

    def tracking_add(*args, **kwargs):
        op_order.append("add")
        return orig_add(*args, **kwargs)

    def tracking_del(*args, **kwargs):
        op_order.append("delete")
        return orig_del(*args, **kwargs)

    with patch.object(manager.vector_store, "add_documents", side_effect=tracking_add), \
         patch.object(manager.vector_store, "delete", side_effect=tracking_del):
        with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_v2])):
            report = manager.sync_directory()

    assert report.replaced == 1
    assert "add" in op_order
    assert "delete" in op_order
    assert op_order.index("add") < op_order.index("delete"), "Novos chunks devem ser adicionados ANTES de deletar os antigos!"


def test_sync_directory_modified_file_embedding_failure_preserves_old(make_manager, counting_embeddings, tmp_path):
    """5b. Falha injetada no embedding durante modificação preserva versão antiga no Chroma e mantém manifesto intacto."""
    manager = make_manager()

    doc_file = tmp_path / "base" / "arquivo.pdf"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_bytes(b"%PDF-1.4 initial content")

    doc_v1 = Document(
        page_content="Versao original estável.",
        metadata={"file_name": "arquivo.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_v1])):
        manager.sync_directory()

    initial_vector_count = manager.get_total_vectors()
    manifest_before = IndexManifest.load(manager.manifest_path)

    # Modifica arquivo e ativa falha no embedding
    doc_file.write_bytes(b"%PDF-1.4 altered content")
    doc_v2 = Document(
        page_content="Conteudo novo que vai falhar ao vetorizar.",
        metadata={"file_name": "arquivo.pdf", "page": 1, "source": str(doc_file)}
    )

    counting_embeddings.fail_on_embed = True

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc_v2])):
        with pytest.raises(RuntimeError, match="Falha injetada no modelo de embeddings"):
            manager.sync_directory()

    # Chunks antigos continuam no Chroma e manifesto permanece o mesmo
    assert manager.get_total_vectors() == initial_vector_count
    manifest_after = IndexManifest.load(manager.manifest_path)
    assert manifest_after.files == manifest_before.files


def test_sync_directory_orphan_deleted_only_with_prune(make_manager, tmp_path):
    """6. Órfão é apagado apenas com a trava liberada (--prune)."""
    manager = make_manager()

    f1 = tmp_path / "base" / "f1.pdf"
    f2 = tmp_path / "base" / "f2.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 file 1")
    f2.write_bytes(b"%PDF-1.4 file 2")

    doc1 = Document(page_content="Texto 1", metadata={"file_name": "f1.pdf", "page": 1, "source": str(f1)})
    doc2 = Document(page_content="Texto 2", metadata={"file_name": "f2.pdf", "page": 1, "source": str(f2)})

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1, doc2])):
        manager.sync_directory()

    # Remove f2 do disco. Resta apenas f1.
    f2.unlink()

    # Com 2 docs, deletar 1 = 50% (> 20%). Sem prune=True, recusa!
    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1])):
        rep_refused = manager.sync_directory(prune=False)
        assert rep_refused.pruned == 0
        assert rep_refused.prune_refused == 1

        # Com prune=True, autoriza a exclusão
        rep_pruned = manager.sync_directory(prune=True)
        assert rep_pruned.pruned == 1
        assert rep_pruned.prune_refused == 0


def test_sync_directory_empty_base_dir_full_manifest_does_not_prune_without_flag(make_manager, tmp_path):
    """7. base/ vazia com manifesto cheio NÃO apaga nada sem prune."""
    manager = make_manager()

    f1 = tmp_path / "base" / "f1.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 single file")

    doc1 = Document(page_content="Texto doc", metadata={"file_name": "f1.pdf", "page": 1, "source": str(f1)})

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1])):
        manager.sync_directory()

    total_before = manager.get_total_vectors()
    assert total_before > 0
    f1.unlink()

    # base/ agora vazia, sincroniza sem prune
    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[])):
        report = manager.sync_directory(prune=False)

        assert report.pruned == 0
        assert report.prune_refused > 0
        assert manager.get_total_vectors() == total_before


def test_sync_directory_unreadable_file_protected(make_manager, tmp_path):
    """8. Arquivo ilegível (em failed_files) é protegido e nunca apagado."""
    manager = make_manager()

    f1 = tmp_path / "base" / "corrompido.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 originally valid")

    doc1 = Document(page_content="Texto valido", metadata={"file_name": "corrompido.pdf", "page": 1, "source": str(f1)})

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1])):
        manager.sync_directory()

    initial_total = manager.get_total_vectors()

    # Próxima sincronização: arquivo falha na leitura (ex: erro de I/O)
    with patch.object(
        manager.loader,
        "load_documents",
        return_value=LoadResult(documents=[], failed_files=[str(f1)])
    ):
        report = manager.sync_directory(prune=True)

        assert report.protected == 1
        assert report.pruned == 0
        assert manager.get_total_vectors() == initial_total


def test_sync_directory_fingerprint_change_triggers_total_reindex(make_manager, tmp_path):
    """9. Mudança de fingerprint (ex: chunk_size) reindexa tudo."""
    manager = make_manager()

    f1 = tmp_path / "base" / "doc.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 file content")

    doc = Document(page_content="Texto para reindexar", metadata={"file_name": "doc.pdf", "page": 1, "source": str(f1)})

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        manager.sync_directory()

    # Altera configuração (chunk_size)
    manager.settings.chunk_size = 500
    manager.splitter.chunk_size = 500

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        report = manager.sync_directory()

        # Deve disparar reindexação total
        assert report.indexed == 1
        manifest = IndexManifest.load(manager.manifest_path)
        assert manifest.fingerprint["chunk_size"] == 500


def test_sync_directory_manifest_chroma_divergence_triggers_total_reindex(make_manager, tmp_path):
    """10. Divergência entre manifesto e ChromaDB dispara reindexação total."""
    manager = make_manager()

    f1 = tmp_path / "base" / "doc.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 doc content")

    doc = Document(page_content="Texto de integridade", metadata={"file_name": "doc.pdf", "page": 1, "source": str(f1)})

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        manager.sync_directory()

    # Força divergência: deleta vetores do Chroma diretamente sem atualizar o manifesto
    all_ids = manager.get_all_chunk_ids()
    manager.vector_store.delete(ids=list(all_ids))
    assert manager.get_total_vectors() == 0

    # Próxima sincronização detecta divergência e reindexa tudo
    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc])):
        report = manager.sync_directory()

        assert report.indexed == 1
        assert manager.get_total_vectors() > 0


def test_sync_directory_positional_argument_raises_type_error(make_manager):
    """sync_directory deve rejeitar argumentos posicionais como caminho de diretório (apenas keyword-only prune)."""
    manager = make_manager()
    with pytest.raises(TypeError):
        manager.sync_directory("base")


def test_criar_db_calls_sync_directory_without_positionals():
    """Valida que criar_db() chama sync_directory sem argumentos posicionais e com prune=False."""
    import criar_db
    with patch("criar_db.VectorStoreManager") as mock_cls:
        mock_mgr = MagicMock()
        mock_cls.return_value = mock_mgr

        criar_db.criar_db()

        mock_cls.assert_called_once()
        mock_mgr.sync_directory.assert_called_once_with(prune=False)
        call_args = mock_mgr.sync_directory.call_args
        assert len(call_args.args) == 0
        assert call_args.kwargs == {"prune": False}


def test_vector_store_smoke_test_empty_dir_offline(tmp_path):
    """5. Teste de fumaça sem API: VectorStoreManager com DeterministicFakeEmbedding e pasta vazia."""
    empty_docs_dir = tmp_path / "empty_base"
    empty_docs_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = tmp_path / "chroma_smoke"
    manifest_path = tmp_path / "smoke_manifest.json"

    settings = Settings(
        documents_directory=str(empty_docs_dir),
        chroma_persist_directory=str(chroma_dir),
        collection_name=f"smoke_col_{uuid.uuid4().hex}",
    )
    embeddings = DeterministicFakeEmbedding(size=128)

    manager = VectorStoreManager(
        embeddings=embeddings,
        settings=settings,
        manifest_path=manifest_path,
    )

    report = manager.sync_directory()

    assert isinstance(report, SyncReport)
    assert report.indexed == 0
    assert report.embedding_calls == 0

    # Acessa manager.vector_store
    vs = manager.vector_store
    assert vs is not None
    assert manager.get_total_vectors() == 0


def test_sync_directory_total_reindex_queries_new_collection(make_manager, tmp_path):
    """Depois de sync_directory com reindexação total, manager.vector_store consulta a coleção nova."""
    manager = make_manager()

    doc_file = tmp_path / "base" / "doc.pdf"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_bytes(b"%PDF-1.4 initial content")

    doc1 = Document(
        page_content="Conteudo antes da reindexacao.",
        metadata={"file_name": "doc.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc1])):
        manager.sync_directory()

    res_before = manager.vector_store.similarity_search("antes", k=1)
    assert len(res_before) == 1
    assert "antes" in res_before[0].page_content

    # Dispara reindexação total alterando o fingerprint (chunk_size)
    manager.settings.chunk_size = 500
    manager.splitter.chunk_size = 500

    doc2 = Document(
        page_content="Conteudo novo totalmente recem-indexado.",
        metadata={"file_name": "doc.pdf", "page": 1, "source": str(doc_file)}
    )

    with patch.object(manager.loader, "load_documents", return_value=LoadResult(documents=[doc2])):
        report = manager.sync_directory()
        assert report.indexed == 1

    # manager.vector_store deve consultar a coleção NOVA com os chunks recém-indexados
    res_after = manager.vector_store.similarity_search("recem-indexado", k=1)
    assert len(res_after) == 1
    assert "Conteudo novo totalmente recem-indexado." in res_after[0].page_content


