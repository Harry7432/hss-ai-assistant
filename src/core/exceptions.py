"""Hierarquia de exceções de domínio do HSS AI Assistant."""


class HSSAssistantError(Exception):
    """Exceção base para todas as falhas de domínio do assistente."""
    pass


class ConfigurationError(HSSAssistantError):
    """Lançada quando parâmetros de configuração são inválidos ou ausentes."""
    pass


class IngestionError(HSSAssistantError):
    """Lançada durante falhas no carregamento ou processamento de PDFs."""
    pass


class VectorStoreError(HSSAssistantError):
    """Lançada quando ocorrem erros no ChromaDB ou persistência vetorial."""
    pass


class RetrievalError(HSSAssistantError):
    """Lançada quando ocorrem falhas durante a busca vetorial."""
    pass


class GenerationError(HSSAssistantError):
    """Lançada quando a geração de resposta falha."""
    pass


class InsufficientContextError(HSSAssistantError):
    """Lançada ou sinalizada quando o contexto não atinge o limiar de relevância."""
    pass


class CitationValidationError(HSSAssistantError):
    """Lançada quando referências ou citações geradas são inconsistentes."""
    pass
