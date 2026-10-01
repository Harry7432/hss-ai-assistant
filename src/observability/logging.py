import logging
import re
from typing import Any, Dict


# Expressões regulares para detecção de chaves e dados sensíveis
API_KEY_REGEX = re.compile(r"sk-[a-zA-Z0-9_\-]{8,}")
BEARER_TOKEN_REGEX = re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE)


def sanitize_log_message(message: str) -> str:
    """Mascara chaves de API e tokens confidenciais em strings de log.

    Args:
        message: Texto bruto a ser gravado em log.

    Returns:
        Texto sanitizado com chaves mascaradas como 'sk-***'.
    """
    if not isinstance(message, str):
        message = str(message)

    sanitized = API_KEY_REGEX.sub("sk-***", message)
    sanitized = BEARER_TOKEN_REGEX.sub("Bearer ***", sanitized)
    return sanitized


class SanitizedFormatter(logging.Formatter):
    """Formatador de log com sanitização automática ativa."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return sanitize_log_message(original)


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado com filtro de sanitização."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = SanitizedFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
