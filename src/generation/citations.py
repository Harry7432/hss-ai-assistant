import re
from dataclasses import dataclass
from typing import List, Tuple
from src.retrieval.search import RetrievedChunk


@dataclass
class Citation:
    """Estrutura formal de uma citação de fonte."""

    marker: str
    file_name: str
    page: int
    snippet: str
    score: float


class CitationFormatter:
    """Formatador e validador programático de citações RAG."""

    def build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        """Gera objetos de citação correspondentes à lista ordenada de chunks."""
        citations: List[Citation] = []
        for idx, chunk in enumerate(chunks, 1):
            file_name = chunk.metadata.get("file_name", "documento.pdf")
            page = chunk.metadata.get("page", 1)
            # Snippet com no máximo 150 caracteres para exibição clara
            clean_content = " ".join(chunk.content.split())
            snippet = (
                clean_content[:140] + "..." if len(clean_content) > 140 else clean_content
            )

            citations.append(
                Citation(
                    marker=f"[{idx}]",
                    file_name=file_name,
                    page=page,
                    snippet=snippet,
                    score=chunk.score,
                )
            )
        return citations

    def format_sources_block(self, citations: List[Citation]) -> str:
        """Renderiza o bloco explicativo de fontes para o rodapé da resposta."""
        if not citations:
            return ""

        lines = ["\n\n### Fontes Consultadas:"]
        for cit in citations:
            lines.append(f"{cit.marker} {cit.file_name} (pág. {cit.page}) - \"{cit.snippet}\"")
        return "\n".join(lines)

    def validate_response_citations(
        self, text: str, valid_markers: List[str]
    ) -> Tuple[str, bool]:
        """Valida que todos os marcadores [N] presentes no texto correspondem a fontes reais.

        Remove referências a marcadores inexistentes/fantasmas gerados pelo LLM.

        Returns:
            Tupla (texto_sanitizado, is_valido).
        """
        all_markers_found = re.findall(r"\[\d+\]", text)
        is_valid = True
        cleaned_text = text

        for marker in all_markers_found:
            if marker not in valid_markers:
                is_valid = False
                # Remove o marcador inválido e eventuais espaços duplos residuais
                cleaned_text = cleaned_text.replace(marker, "").replace("  ", " ")

        return cleaned_text.strip(), is_valid
