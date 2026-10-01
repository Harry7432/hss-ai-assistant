import json
from pathlib import Path
from unittest.mock import patch
import pytest
from src.generation.rag_engine import RAGEngine
from src.retrieval.search import RetrievedChunk


pytestmark = pytest.mark.eval


@pytest.fixture
def eval_dataset():
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.eval
def test_rag_quality_benchmarks(in_memory_chroma, mock_llm_factory, eval_dataset):
    """Executa a suíte de avaliação de qualidade do RAG (RAG Eval) sobre o dataset anotado."""
    # Prepara LLM simulado com respostas fundamentadas
    llm = mock_llm_factory([
        "O vídeo aborda fundamentos de Python, variáveis e laços [1].",
        "Para acessar o material use o link da descrição [1].",
        "Não foi possível encontrar informações suficientes.",
        "Não foi possível encontrar informações suficientes."
    ])

    engine = RAGEngine(vector_store=in_memory_chroma, llm=llm)

    correct_rejections = 0
    correct_answers = 0

    for item in eval_dataset:
        question = item["question"]
        is_unanswerable = item["is_unanswerable"]

        if is_unanswerable:
            # Caso não respondível: sem chunks recuperados acima do threshold
            with patch.object(engine.retriever, "search", return_value=[]):
                response = engine.query(question)
                assert response.is_fallback is True, f"Esperava recusa para '{question}'"
                assert "Não foi possível encontrar" in response.answer
                assert len(response.citations) == 0
                correct_rejections += 1
        else:
            # Caso respondível: simula recuperação do chunk correto
            chunk = RetrievedChunk(
                chunk_id=f"chk_{item['id']}",
                content=f"Conteúdo informativo sobre {item['expected_context_keywords'][0]} e aulas.",
                metadata={"file_name": item["expected_source_file"], "page": 1},
                score=0.92
            )

            with patch.object(engine.retriever, "search", return_value=[chunk]):
                response = engine.query(question)
                assert response.is_fallback is False, f"Esperava resposta para '{question}'"
                assert len(response.citations) > 0, "Esperava citações geradas"
                assert response.citations[0].file_name == item["expected_source_file"]
                assert "[1]" in response.answer
                correct_answers += 1

    # Validação das métricas globais de qualidade
    assert correct_rejections == 2, "100% de recusas corretas para perguntas fora do domínio"
    assert correct_answers == 2, "100% de respostas assertivas com citações válidas"
