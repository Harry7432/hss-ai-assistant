import pytest
from src.memory.session_store import SessionStore


def test_session_store_add_and_retrieve():
    """Valida armazenamento e recuperação de turnos de diálogo."""
    store = SessionStore(max_turns=5)
    store.add_turn("sessao_1", "Qual o tema do curso?", "O tema é Python.")

    history = store.get_history("sessao_1")
    assert len(history) == 1
    assert history[0]["user"] == "Qual o tema do curso?"
    assert history[0]["assistant"] == "O tema é Python."


def test_session_store_segregation():
    """Valida que sessões diferentes não compartilham histórico."""
    store = SessionStore()
    store.add_turn("sessao_A", "Pergunta A", "Resposta A")
    store.add_turn("sessao_B", "Pergunta B", "Resposta B")

    assert len(store.get_history("sessao_A")) == 1
    assert store.get_history("sessao_A")[0]["user"] == "Pergunta A"

    assert len(store.get_history("sessao_B")) == 1
    assert store.get_history("sessao_B")[0]["user"] == "Pergunta B"


def test_session_store_sliding_window():
    """Valida controle de janela deslizante (limite de turnos)."""
    store = SessionStore(max_turns=2)
    store.add_turn("sessao_1", "P1", "R1")
    store.add_turn("sessao_1", "P2", "R2")
    store.add_turn("sessao_1", "P3", "R3")

    history = store.get_history("sessao_1")
    assert len(history) == 2
    assert history[0]["user"] == "P2"
    assert history[1]["user"] == "P3"
