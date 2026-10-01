from typing import Dict, List


class SessionStore:
    """Repositório de memória conversacional multi-turn em memória com janela deslizante."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Adiciona um turno de diálogo à sessão informada."""
        history = self._sessions.setdefault(session_id, [])
        history.append({"user": user_message, "assistant": assistant_message})

        # Aplicar janela deslizante se ultrapassar max_turns
        if len(history) > self.max_turns:
            self._sessions[session_id] = history[-self.max_turns :]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retorna o histórico de mensagens da sessão."""
        return list(self._sessions.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Limpa o histórico de uma sessão específica."""
        if session_id in self._sessions:
            del self._sessions[session_id]
