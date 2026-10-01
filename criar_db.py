"""Script legado para criação e sincronização do banco de dados vetorial.

Mantido para compatibilidade, delegando a execução para a arquitetura modular em src.indexing.
"""
from src.indexing.vector_store import VectorStoreManager


def criar_db(pasta_base: str = "base") -> dict:
    """Sincroniza os documentos da pasta base com o ChromaDB de forma idempotente."""
    manager = VectorStoreManager()
    resultado = manager.sync_directory(pasta_base)
    print("Banco de Dados sincronizado com sucesso!")
    print(f"Resumo da sincronização: {resultado}")
    return resultado


if __name__ == "__main__":
    criar_db()