import logging
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

RECONTEXT_PROMPT_TEMPLATE = """Com base no histórico da conversa abaixo e na nova pergunta do usuário, reformule a pergunta para que ela se torne uma consulta de busca independente e autocontida.
NÃO responda à pergunta, apenas retorne a pergunta reformulada. Se a pergunta já for independente ou não depender do histórico, repita a pergunta original exatamente como foi escrita.

Histórico da Conversa:
{history_text}

Nova Pergunta do Usuário:
{query}

Pergunta Reformulada Autocontida:"""


class QueryRecontextualizer:
    """Reescreve perguntas dependentes de contexto anterior em queries autocontidas."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.0,
                openai_api_key=self.settings.openai_api_key,
            )

    def recontextualize(self, query: str, history: List[Dict[str, str]]) -> str:
        """Gera versão autocontida da consulta se houver histórico prévio."""
        clean_query = query.strip() if query else ""
        if not history or not clean_query:
            return clean_query

        # Montar histórico resumido
        dialogue_lines: List[str] = []
        for turn in history[-4:]:  # Usa no máximo os 4 últimos turnos
            u = turn.get("user", "")
            a = turn.get("assistant", "")
            dialogue_lines.append(f"Usuário: {u}\nAssistente: {a}")

        history_text = "\n\n".join(dialogue_lines)
        prompt = RECONTEXT_PROMPT_TEMPLATE.format(
            history_text=history_text, query=clean_query
        )

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response)).strip()
            logger.debug(f"Query recontextualizada: '{clean_query}' -> '{content}'")
            return content if content else clean_query
        except Exception as e:
            logger.warning(
                f"Falha na recontextualização da query ({e}); mantendo query original."
            )
            return clean_query
