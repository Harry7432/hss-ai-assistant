"""Script legado para criação e sincronização do banco de dados vetorial.

Mantido para compatibilidade, delegando a execução para a arquitetura modular em src.indexing.
"""
from src.indexing.vector_store import VectorStoreManager


def criar_db() -> dict:
    """Sincroniza os documentos da pasta configurada com o ChromaDB de forma idempotente."""
    manager = VectorStoreManager()
    resultado = manager.sync_directory(prune=False)
    print("Banco de Dados sincronizado com sucesso!")
    print(f"Resumo da sincronização: {resultado}")
    return resultado


if __name__ == "__main__":
    criar_db()