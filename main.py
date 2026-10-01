"""Ponto de entrada do HSS AI Assistant.

Mantido para compatibilidade retroativa, delegando para a arquitetura modular em src.
"""
from typing import Optional
from src.core.config import get_settings
from src.indexing.vector_store import VectorStoreManager
from src.generation.rag_engine import RAGEngine
from src.cli.chat_loop import run_chat_loop


def perguntar(pergunta_texto: Optional[str] = None):
    """Executa uma consulta pontual utilizando o motor RAG modular."""
    settings = get_settings()
    manager = VectorStoreManager(settings=settings)
    engine = RAGEngine(vector_store=manager.vector_store, settings=settings)

    if pergunta_texto is None:
        pergunta_texto = input("Escreva sua pergunta: ")

    resposta = engine.query(pergunta_texto)
    print("Resposta da IA:")
    print(resposta.answer)
    return resposta


if __name__ == "__main__":
    run_chat_loop()