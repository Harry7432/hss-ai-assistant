from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config import Settings, get_settings
from src.indexing.splitter import SPLITTER_VERSION

logger = logging.getLogger(__name__)

FINGERPRINT_FIELDS = [
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "splitter_version",
    "distance_metric",
    "collection_name",
]


def build_fingerprint(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Constrói o dicionário de fingerprint a partir das configurações ativas."""
    s = settings or get_settings()
    return {
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "splitter_version": SPLITTER_VERSION,
        "distance_metric": s.distance_metric,
        "collection_name": s.collection_name,
    }


@dataclass
class ManifestFileEntry:
    """Entrada de metadados de arquivo indexado no manifesto."""
    paths: List[str]
    file_name: str
    chunk_ids: List[str]
    indexed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": list(self.paths),
            "file_name": self.file_name,
            "chunk_ids": list(self.chunk_ids),
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestFileEntry":
        return cls(
            paths=list(data.get("paths", [])),
            file_name=str(data.get("file_name", "")),
            chunk_ids=list(data.get("chunk_ids", [])),
            indexed_at=str(data.get("indexed_at", "")),
        )


@dataclass
class IndexManifest:
    """Manifesto de integridade e rastreabilidade do índice vetorial."""
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_fingerprint_valid(self, expected_fingerprint: Dict[str, Any]) -> bool:
        """Verifica se todos os 6 campos do fingerprint coincidem rigorosamente com o esperado."""
        if not self.fingerprint:
            return False
        for field_name in FINGERPRINT_FIELDS:
            if field_name not in self.fingerprint or field_name not in expected_fingerprint:
                return False
            if self.fingerprint[field_name] != expected_fingerprint[field_name]:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": dict(self.fingerprint),
            "files": dict(self.files),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexManifest":
        return cls(
            fingerprint=dict(data.get("fingerprint", {})),
            files=dict(data.get("files", {})),
        )

    def save_atomic(self, target_path: Path | str) -> None:
        """Gravação atômica: escreve em arquivo temporário e substitui com os.replace."""
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_file = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, dest)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logger.error(f"Erro na gravação atômica do manifesto em {dest}: {e}")
            raise

    @classmethod
    def load(cls, manifest_path: Path | str) -> "IndexManifest":
        """Carrega manifesto do disco. Se ausente ou corrompido, emite warning e retorna manifesto vazio."""
        path = Path(manifest_path)
        if not path.exists():
            logger.warning(f"Manifesto não encontrado em {path}. Retornando manifesto vazio.")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"Manifesto em {path} possui formato inválido. Tratando como vazio.")
                return cls()
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Manifesto corrompido ou ilegível em {path}: {e}. Tratando como vazio.")
            return cls()


def reconcile_manifest_with_chroma(manifest: IndexManifest, chroma_chunk_ids: Set[str]) -> bool:
    """Compara os chunk_ids do manifesto com os IDs existentes no Chroma.

    Divergência em qualquer sentido = reindexação total.
    - True: conjuntos iguais.
    - False: divergência (IDs faltando no Chroma ou IDs sobrando no Chroma).
    """
    manifest_chunk_ids: Set[str] = set()
    for file_entry in manifest.files.values():
        chunk_ids = file_entry.get("chunk_ids", [])
        manifest_chunk_ids.update(chunk_ids)

    if manifest_chunk_ids == chroma_chunk_ids:
        return True

    missing_in_chroma = manifest_chunk_ids - chroma_chunk_ids
    extra_in_chroma = chroma_chunk_ids - manifest_chunk_ids

    if missing_in_chroma:
        logger.warning(
            f"Divergência na reconciliação: {len(missing_in_chroma)} chunk(s) do manifesto ausentes no Chroma. "
            f"Exemplos: {list(missing_in_chroma)[:3]}"
        )
    if extra_in_chroma:
        logger.warning(
            f"Divergência na reconciliação: {len(extra_in_chroma)} chunk(s) no Chroma não listados no manifesto. "
            f"Exemplos: {list(extra_in_chroma)[:3]}"
        )

    return False


@dataclass
class RenameAction:
    """Ação de atualização de metadados para arquivo renomeado."""
    doc_hash: str
    old_paths: List[str]
    new_path: str
    file_name: str


@dataclass
class SyncPlan:
    """Plano de sincronização puro gerado a partir do estado em disco e do manifesto."""
    unchanged: List[str] = field(default_factory=list)
    to_index: Dict[str, str] = field(default_factory=dict)
    renamed: List[RenameAction] = field(default_factory=list)
    ignored_copies: List[str] = field(default_factory=list)
    replaced: Set[str] = field(default_factory=set)
    orphans: Set[str] = field(default_factory=set)
    protected: Set[str] = field(default_factory=set)
    total_manifest_docs: int = 0


def create_sync_plan(
    files_on_disk: Dict[str, str],
    failed_files: Set[str],
    manifest: IndexManifest,
) -> SyncPlan:
    """Gera um plano de sincronização determinístico sem dependência do ChromaDB.

    Args:
        files_on_disk: Mapeamento de caminho -> doc_hash dos arquivos lidos no disco.
        failed_files: Conjunto de caminhos de arquivos que falharam na leitura.
        manifest: Instância atual de IndexManifest.

    Returns:
        SyncPlan categorizando cada situação.
    """
    total_docs = len(manifest.files)
    plan = SyncPlan(total_manifest_docs=total_docs)

    # 1. Identificar arquivos protegidos (falharam na leitura - NUNCA viram órfãos)
    for doc_hash, entry in manifest.files.items():
        m_paths = entry.get("paths", [])
        if any(p in failed_files for p in m_paths):
            plan.protected.add(doc_hash)
            logger.warning(
                f"Arquivo com falha de leitura protegido contra expurgo: {doc_hash} (caminhos: {m_paths})"
            )

    # Mapeamento invertido: caminho no manifesto -> doc_hash registrado
    manifest_path_to_hash: Dict[str, str] = {}
    for doc_hash, entry in manifest.files.items():
        for p in entry.get("paths", []):
            manifest_path_to_hash[p] = doc_hash

    # 2. Avaliar cada arquivo encontrado no disco
    for disk_path, disk_hash in files_on_disk.items():
        if disk_hash in manifest.files:
            entry = manifest.files[disk_hash]
            known_paths = entry.get("paths", [])
            if disk_path in known_paths:
                # Mesmo caminho e mesmo hash
                plan.unchanged.append(disk_path)
            else:
                # Hash já indexado, mas caminho novo
                old_paths_exist = any(p in files_on_disk for p in known_paths)
                if old_paths_exist:
                    # Caminho antigo ainda existe no disco -> cópia ignorada com warning
                    logger.warning(
                        f"Cópia duplicada detectada e ignorada: {disk_path} tem o mesmo hash de {known_paths}"
                    )
                    plan.ignored_copies.append(disk_path)
                else:
                    # Caminho antigo AUSENTE do disco e novo presente -> renomeação!
                    file_name = Path(disk_path).name
                    plan.renamed.append(
                        RenameAction(
                            doc_hash=disk_hash,
                            old_paths=list(known_paths),
                            new_path=disk_path,
                            file_name=file_name,
                        )
                    )
        else:
            # Hash novo (arquivo novo ou modificado)
            plan.to_index[disk_path] = disk_hash
            # Se o caminho já constava no manifesto com outro hash, é modificado (replaced)
            if disk_path in manifest_path_to_hash:
                old_hash = manifest_path_to_hash[disk_path]
                plan.replaced.add(old_hash)

    # 3. Identificar órfãos
    renamed_hashes = {r.doc_hash for r in plan.renamed}
    for doc_hash, entry in manifest.files.items():
        if doc_hash in plan.protected:
            continue
        if doc_hash in plan.replaced:
            # Arquivo modificado: vetores antigos saem depois (Fatia 4)
            continue
        if doc_hash in renamed_hashes:
            continue

        known_paths = entry.get("paths", [])
        # Se nenhum dos caminhos conhecidos estiver no disco (nem inalterado, nem com outro hash)
        if not any(p in files_on_disk for p in known_paths):
            plan.orphans.add(doc_hash)

    return plan


def check_prune_safety(
    plan: SyncPlan,
    max_prune_percentage: int,
    prune: bool,
) -> Tuple[bool, str]:
    """Verifica a trava de segurança na remoção de órfãos.

    Percentual = orphans dividido pelo total de documentos do manifesto.
    Acima do limite e sem a flag: recusar e retornar aviso de segurança.
    Com um único documento no manifesto, qualquer exclusão exige a flag.
    """
    total_docs = plan.total_manifest_docs
    orphans_count = len(plan.orphans)

    if orphans_count == 0:
        return True, "Nenhum documento órfão a podar."

    # Com um único documento no manifesto, qualquer exclusão exige a flag prune
    if total_docs <= 1:
        if not prune:
            return False, "Exclusão do único documento do índice exige a flag --prune."
        return True, "Poda autorizada pela flag --prune."

    percentage = (orphans_count / total_docs) * 100.0
    if percentage > max_prune_percentage:
        if not prune:
            return (
                False,
                f"Poda abortada: percentual de órfãos ({percentage:.1f}%) excede o limite "
                f"de segurança ({max_prune_percentage}%). Use a flag --prune para autorizar.",
            )
        return True, f"Poda autorizada pela flag --prune ({percentage:.1f}% de órfãos)."

    return True, f"Poda segura dentro do limite ({percentage:.1f}% <= {max_prune_percentage}%)."
