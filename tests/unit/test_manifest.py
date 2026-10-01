import json
import logging
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from src.core.config import Settings, get_settings
from src.indexing.splitter import SPLITTER_VERSION
from src.indexing.manifest import (
    IndexManifest,
    ManifestFileEntry,
    SyncPlan,
    RenameAction,
    build_fingerprint,
    create_sync_plan,
    reconcile_manifest_with_chroma,
    check_prune_safety,
    FINGERPRINT_FIELDS,
)


def test_settings_collection_name_and_distance_metric_and_splitter_version():
    """Valida configurações de collection_name, distance_metric fixo e SPLITTER_VERSION."""
    settings = Settings()
    assert settings.collection_name == "hss_docs"
    assert settings.distance_metric == "cosine"
    assert SPLITTER_VERSION == "1"


def test_manifest_structure_and_serialization():
    """Valida estrutura de IndexManifest com os 6 campos no fingerprint e arquivos por doc_hash."""
    fp = {
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "splitter_version": "1",
        "distance_metric": "cosine",
        "collection_name": "hss_docs",
    }
    files = {
        "hash_abc_123": {
            "paths": ["base/manual.pdf"],
            "file_name": "manual.pdf",
            "chunk_ids": ["hash_abc_123_0", "hash_abc_123_1"],
            "indexed_at": "2026-10-01T00:00:00Z",
        }
    }
    manifest = IndexManifest(fingerprint=fp, files=files)
    data = manifest.to_dict()

    assert data["fingerprint"] == fp
    assert data["files"] == files

    restored = IndexManifest.from_dict(data)
    assert restored.fingerprint == fp
    assert restored.files == files
    assert restored.is_fingerprint_valid(fp) is True


def test_manifest_atomic_save_and_recovery_on_failure(tmp_path):
    """Valida que a gravação é atômica: se falhar no meio, o arquivo anterior permanece íntegro."""
    manifest_file = tmp_path / "manifest.json"
    
    # Cria versão inicial íntegra
    initial_fp = {"chunk_size": 500, "collection_name": "initial"}
    initial_files = {"hash_original": {"paths": ["base/doc.pdf"], "file_name": "doc.pdf", "chunk_ids": [], "indexed_at": ""}}
    initial_manifest = IndexManifest(fingerprint=initial_fp, files=initial_files)
    initial_manifest.save_atomic(manifest_file)

    assert manifest_file.exists()
    content_before = manifest_file.read_text(encoding="utf-8")

    # Tenta salvar nova versão, mas simula falha durante a escrita do arquivo temporário
    new_manifest = IndexManifest(fingerprint={"chunk_size": 1000}, files={})
    
    with patch("json.dump", side_effect=IOError("Simulação de disco cheio")):
        with pytest.raises(IOError, match="Simulação de disco cheio"):
            new_manifest.save_atomic(manifest_file)

    # O arquivo original DEVE continuar intacto e inalterado
    assert manifest_file.exists()
    assert manifest_file.read_text(encoding="utf-8") == content_before


def test_manifest_load_missing_or_corrupted(tmp_path, caplog):
    """Valida que arquivo ausente ou corrompido emite warning e retorna manifesto vazio sem exceção."""
    # 1. Arquivo ausente
    missing_file = tmp_path / "inexistente.json"
    with caplog.at_level(logging.WARNING):
        m1 = IndexManifest.load(missing_file)
    assert isinstance(m1, IndexManifest)
    assert m1.files == {}
    assert m1.fingerprint == {}
    assert any("não encontrado" in r.message.lower() for r in caplog.records)

    # 2. Arquivo corrompido (JSON inválido)
    corrupted_file = tmp_path / "corrupted.json"
    corrupted_file.write_text("{conteudo corrompido que nao e json válido", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        m2 = IndexManifest.load(corrupted_file)
    assert isinstance(m2, IndexManifest)
    assert m2.files == {}
    assert m2.fingerprint == {}
    assert any("corrompido" in r.message.lower() or "ilegível" in r.message.lower() for r in caplog.records)


def test_fingerprint_invalidation_each_of_the_6_fields():
    """Valida que a alteração de qualquer um dos 6 campos do fingerprint invalida o índice."""
    base_fp = {
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "splitter_version": "1",
        "distance_metric": "cosine",
        "collection_name": "hss_docs",
    }
    manifest = IndexManifest(fingerprint=base_fp)

    # Todos os 6 campos devem estar previstos
    assert set(FINGERPRINT_FIELDS) == {
        "embedding_model", "chunk_size", "chunk_overlap",
        "splitter_version", "distance_metric", "collection_name"
    }

    # Fingerprint idêntico é válido
    assert manifest.is_fingerprint_valid(base_fp) is True

    # Modificar cada um dos 6 campos deve invalidar
    variations = {
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "splitter_version": "2",
        "distance_metric": "l2",
        "collection_name": "outra_colecao",
    }

    for field_name, altered_value in variations.items():
        altered_fp = dict(base_fp)
        altered_fp[field_name] = altered_value
        assert manifest.is_fingerprint_valid(altered_fp) is False, (
            f"Esperava que alteração em '{field_name}' invalidasse o fingerprint."
        )

        # Campo ausente também invalida
        missing_fp = dict(base_fp)
        del missing_fp[field_name]
        assert manifest.is_fingerprint_valid(missing_fp) is False, (
            f"Esperava que ausência de '{field_name}' invalidasse o fingerprint."
        )


def test_reconciliation_with_chroma_equal_missing_and_extra_ids(caplog):
    """Valida reconciliação entre chunk_ids do manifesto e ChromaDB (conjuntos iguais, faltando e sobrando)."""
    manifest = IndexManifest(
        files={
            "hash_1": {"chunk_ids": ["h1_0", "h1_1"], "paths": ["doc1.pdf"]},
            "hash_2": {"chunk_ids": ["h2_0"], "paths": ["doc2.pdf"]},
        }
    )
    expected_ids = {"h1_0", "h1_1", "h2_0"}

    # 1. Conjuntos iguais -> Sucesso
    assert reconcile_manifest_with_chroma(manifest, expected_ids) is True

    # 2. Faltando IDs no Chroma (Chroma tem menos que o manifesto) -> Divergência
    missing_in_chroma = {"h1_0", "h1_1"}  # falta h2_0
    with caplog.at_level(logging.WARNING):
        assert reconcile_manifest_with_chroma(manifest, missing_in_chroma) is False
    assert any("ausentes no chroma" in r.message.lower() for r in caplog.records)

    # 3. Sobrando IDs no Chroma (Chroma tem IDs extras não catalogados) -> Divergência
    extra_in_chroma = {"h1_0", "h1_1", "h2_0", "h_desconhecido_99"}
    with caplog.at_level(logging.WARNING):
        assert reconcile_manifest_with_chroma(manifest, extra_in_chroma) is False
    assert any("não listados no manifesto" in r.message.lower() for r in caplog.records)


def test_sync_plan_unchanged_files():
    """Valida identificação de arquivos inalterados (mesmo caminho e mesmo hash)."""
    manifest = IndexManifest(
        files={
            "hash_doc1": {"paths": ["base/doc1.pdf"], "file_name": "doc1.pdf", "chunk_ids": ["c1"]}
        }
    )
    files_on_disk = {"base/doc1.pdf": "hash_doc1"}
    failed_files = set()

    plan = create_sync_plan(files_on_disk, failed_files, manifest)
    assert plan.unchanged == ["base/doc1.pdf"]
    assert plan.to_index == {}
    assert plan.renamed == []
    assert plan.ignored_copies == []
    assert plan.replaced == set()
    assert plan.orphans == set()
    assert plan.protected == set()


def test_sync_plan_to_index_new_and_modified_files():
    """Valida identificação de novos arquivos e arquivos modificados (replaced)."""
    manifest = IndexManifest(
        files={
            "hash_antigo": {"paths": ["base/modificado.pdf"], "file_name": "modificado.pdf", "chunk_ids": ["c1"]}
        }
    )
    files_on_disk = {
        "base/modificado.pdf": "hash_novo",  # mesmo caminho, hash diferente
        "base/novo.pdf": "hash_totalmente_novo",  # novo caminho e novo hash
    }
    failed_files = set()

    plan = create_sync_plan(files_on_disk, failed_files, manifest)
    assert plan.to_index == {
        "base/modificado.pdf": "hash_novo",
        "base/novo.pdf": "hash_totalmente_novo",
    }
    assert plan.replaced == {"hash_antigo"}
    assert plan.orphans == set()  # Modificado vai para replaced, não para orphans
    assert plan.unchanged == []


def test_sync_plan_renamed_file():
    """Valida que arquivo renomeado (caminho antigo ausente, novo presente, mesmo hash) é detectado."""
    manifest = IndexManifest(
        files={
            "hash_arquivo_renomeado": {
                "paths": ["base/nome_antigo.pdf"],
                "file_name": "nome_antigo.pdf",
                "chunk_ids": ["c1", "c2"],
            }
        }
    )
    # No disco, o caminho antigo sumiu e apareceu no novo caminho
    files_on_disk = {"base/nome_novo.pdf": "hash_arquivo_renomeado"}
    failed_files = set()

    plan = create_sync_plan(files_on_disk, failed_files, manifest)
    assert len(plan.renamed) == 1
    rename = plan.renamed[0]
    assert rename.doc_hash == "hash_arquivo_renomeado"
    assert rename.old_paths == ["base/nome_antigo.pdf"]
    assert rename.new_path == "base/nome_novo.pdf"
    assert rename.file_name == "nome_novo.pdf"
    assert plan.to_index == {}
    assert plan.orphans == set()


def test_sync_plan_ignored_copies(caplog):
    """Valida que cópia (caminho antigo ainda presente no disco com mesmo hash) é ignorada com warning."""
    manifest = IndexManifest(
        files={
            "hash_doc": {
                "paths": ["base/original.pdf"],
                "file_name": "original.pdf",
                "chunk_ids": ["c1"],
            }
        }
    )
    # Ambos os caminhos estão presentes no disco
    files_on_disk = {
        "base/original.pdf": "hash_doc",
        "base/copia_de_original.pdf": "hash_doc",
    }
    failed_files = set()

    with caplog.at_level(logging.WARNING):
        plan = create_sync_plan(files_on_disk, failed_files, manifest)

    assert "base/original.pdf" in plan.unchanged
    assert plan.ignored_copies == ["base/copia_de_original.pdf"]
    assert plan.to_index == {}
    assert plan.renamed == []
    assert any("cópia duplicada" in r.message.lower() or "ignorada" in r.message.lower() for r in caplog.records)


def test_sync_plan_orphans_and_protected_files():
    """Valida que arquivos ausentes viram orphans, exceto se estiverem em failed_files (protected)."""
    manifest = IndexManifest(
        files={
            "hash_deletado": {
                "paths": ["base/deletado.pdf"],
                "file_name": "deletado.pdf",
                "chunk_ids": ["c1"],
            },
            "hash_corrompido": {
                "paths": ["base/corrompido.pdf"],
                "file_name": "corrompido.pdf",
                "chunk_ids": ["c2"],
            },
        }
    )
    files_on_disk = {}  # Nenhum arquivo lido com sucesso no disco
    failed_files = {"base/corrompido.pdf"}  # Este arquivo falhou na leitura

    plan = create_sync_plan(files_on_disk, failed_files, manifest)

    # O deletado deve virar orphan
    assert "hash_deletado" in plan.orphans
    # O arquivo com falha de leitura DEVE ser protegido e NUNCA virar orphan
    assert "hash_corrompido" in plan.protected
    assert "hash_corrompido" not in plan.orphans


def test_prune_guard_safety_threshold():
    """Valida a trava de segurança de poda contra expurgo excessivo de órfãos."""
    # Cenário: 10 documentos no total, limite padrão 20%
    # 2 órfãos = 20% (limite aceitável) -> permitido sem prune
    plan_safe = SyncPlan(orphans={"h1", "h2"}, total_manifest_docs=10)
    allowed, msg = check_prune_safety(plan_safe, max_prune_percentage=20, prune=False)
    assert allowed is True

    # 3 órfãos = 30% (> 20%) -> RECUSADO sem a flag prune
    plan_unsafe = SyncPlan(orphans={"h1", "h2", "h3"}, total_manifest_docs=10)
    allowed, msg = check_prune_safety(plan_unsafe, max_prune_percentage=20, prune=False)
    assert allowed is False
    assert "limite de segurança" in msg.lower() or "abortada" in msg.lower()

    # Com a flag prune=True -> PERMITIDO mesmo acima do limite
    allowed, msg = check_prune_safety(plan_unsafe, max_prune_percentage=20, prune=True)
    assert allowed is True

    # Zero órfãos -> sempre permitido
    plan_empty = SyncPlan(orphans=set(), total_manifest_docs=10)
    allowed, msg = check_prune_safety(plan_empty, max_prune_percentage=20, prune=False)
    assert allowed is True


def test_prune_guard_single_document_in_manifest():
    """Valida que com 1 único documento no manifesto, qualquer exclusão exige a flag prune."""
    plan = SyncPlan(orphans={"h1"}, total_manifest_docs=1)

    # Sem a flag prune -> recusado
    allowed, msg = check_prune_safety(plan, max_prune_percentage=20, prune=False)
    assert allowed is False
    assert "único documento" in msg.lower() or "flag --prune" in msg.lower()

    # Com a flag prune -> autorizado
    allowed, msg = check_prune_safety(plan, max_prune_percentage=20, prune=True)
    assert allowed is True
