import os
import logging
from typing import Optional
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def setup_telemetry(settings: Optional[Settings] = None) -> bool:
    """Configura integração com LangSmith para observabilidade condicional.

    Returns:
        True se a telemetria estiver ativa, False caso contrário.
    """
    cfg = settings or get_settings()

    if cfg.langchain_tracing_v2 and cfg.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langchain_project
        logger.info(f"LangSmith Tracing ativado com sucesso para o projeto '{cfg.langchain_project}'.")
        return True

    logger.debug("LangSmith Tracing desativado via configuração.")
    return False
