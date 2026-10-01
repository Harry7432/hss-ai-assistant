import sys
from src.core.config import get_settings
from src.indexing.vector_store import VectorStoreManager
from src.generation.rag_engine import RAGEngine
from src.observability.telemetry import setup_telemetry


def run_chat_loop():
    """Inicia o loop interativo do assistente no terminal."""
    settings = get_settings()
    setup_telemetry(settings)

    print("\n" + "=" * 55)
    print("🤖 HSS AI Assistant - Motor RAG Modular Ativo")
    print("Comandos especiais: 'sair' | 'limpar' | 'sessao'")
    print("=" * 55 + "\n")

    try:
        manager = VectorStoreManager(settings=settings)
        engine = RAGEngine(vector_store=manager.vector_store, settings=settings)
    except Exception as e:
        print(f"❌ Erro ao inicializar o assistente: {e}")
        print("Verifique se as variáveis no arquivo .env estão configuradas corretamente.")
        return

    session_id = "cli_session"

    while True:
        try:
            pergunta = input("\nEscreva sua pergunta: ").strip()
            if not pergunta:
                continue

            if pergunta.lower() in ("sair", "exit", "quit"):
                print("\nEncerrando sessão. Até logo!\n")
                break

            if pergunta.lower() in ("limpar", "clear"):
                engine.session_store.clear_session(session_id)
                print("🧹 Memória da sessão limpa com sucesso.")
                continue

            if pergunta.lower() == "sessao":
                hist = engine.session_store.get_history(session_id)
                print(f"Histórico atual da sessão ({len(hist)} turnos):")
                for idx, t in enumerate(hist, 1):
                    print(f"  {idx}. Q: {t['user']} | A: {t['assistant'][:50]}...")
                continue

            response = engine.query(pergunta, session_id=session_id)

            print("\nResposta da IA:")
            print(response.answer)

            latency = response.metrics.get("latency_ms", 0.0)
            chunks_count = response.metrics.get("chunks_count", 0)
            print(f"\n⏱️ [{latency:.0f}ms | {chunks_count} fontes avaliadas]")

        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário. Encerrando.")
            break
        except Exception as e:
            print(f"❌ Ocorreu um erro ao processar a pergunta: {e}")
