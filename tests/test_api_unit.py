import pytest
import logging
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.config import GEMINI_MODEL, GEMINI_API_KEY
from langchain_core.documents import Document

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint returns success state."""
    with patch("src.vector_db.vector_store") as mock_vector:
        response = client.get("/health")
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "ok"
        assert res["model"] == GEMINI_MODEL
        assert res["platform"] == "Fintech RAG Chatbot"

@patch("src.vector_db.vector_store")
def test_ingest_endpoint(mock_vector_store):
    """Verify document chunking and ingestion workflow."""
    mock_vector_store.add_documents.return_value = None
    response = client.post(
        "/ingest",
        json={
            "text": "This is a public documentation document for the Fintech SaaS platform features.",
            "metadata": {"source": "unit_test"}
        }
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["chunk_count"] > 0
    mock_vector_store.add_documents.assert_called_once()

def test_reciprocal_rank_fusion():
    """Verify correctness of reciprocal rank fusion (RRF) algorithm."""
    from src.rrf import reciprocal_rank_fusion
    
    doc_a = Document(page_content="A", metadata={})
    doc_b = Document(page_content="B", metadata={})
    
    # Dense: doc_a first (0.1), doc_b second (0.4)
    dense_results = [(doc_a, 0.1), (doc_b, 0.4)]
    # Sparse: doc_b first (10.0), doc_a second (2.0)
    sparse_results = [(doc_b, 10.0), (doc_a, 2.0)]
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=2)
    assert len(fused) == 2
    assert fused[0].page_content in ["A", "B"]

@patch("src.main.llm_with_tools")
def test_query_endpoint_refusal(mock_llm_with_tools):
    """Verify that queries that do not trigger tool calls are directly refused."""
    mock_res = MagicMock()
    mock_res.content = "Unrelated response text."
    mock_res.tool_calls = []
    mock_llm_with_tools.invoke.return_value = mock_res
    
    response = client.post(
        "/query",
        json={"message": "What is the capital of France?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "knowledge base" in res["response"]
    assert res["tool_calls_executed"] == []

@patch("src.vector_db.vector_store")
@patch("src.main.llm_with_tools")
def test_query_endpoint_retrieval(mock_llm_with_tools, mock_vector_store):
    """Verify retrieval-based queries and final answer synthesis."""
    # First call: LLM decides to call tool
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "retrieve_local_documents",
        "args": {"query": "wire transfer"},
        "id": "call_99"
    }]
    
    # Second call: LLM synthesizes final answer
    mock_res_answer = MagicMock()
    mock_res_answer.content = "The wire transfer limit is $10,000."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    # Mock Chroma vector database retrieval
    doc = Document(page_content="Wire transfer limits are set to $10,000.", metadata={})
    mock_vector_store.similarity_search_with_score.return_value = [(doc, 0.1)]
    mock_vector_store.get.return_value = {
        "documents": ["Wire transfer limits are set to $10,000."],
        "metadatas": [{}]
    }
    
    response = client.post(
        "/query",
        json={"message": "What is the wire transfer limit?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "10,000" in res["response"]
    assert "retrieve_local_documents" in res["tool_calls_executed"]

@patch("src.main.llm_with_tools")
def test_query_endpoint_invalid_tool_safeguard(mock_llm_with_tools, caplog):
    """Verify invalid tool call safeguarding and fallback triggers."""
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "hallucinated_tool_abc",
        "args": {"query": "test"},
        "id": "call_err"
    }]
    
    # After invalid tool call is intercepted, LLM is re-invoked. It returns direct response
    # which is then overridden by refusal (since no valid tool was executed).
    mock_res_answer = MagicMock()
    mock_res_answer.content = "Fallback generated answer"
    mock_res_answer.tool_calls = []
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/query",
            json={"message": "Query triggering bad tool name", "history": []}
        )
    assert response.status_code == 200
    res = response.json()
    assert res["response"] == "Fallback generated answer"
    assert res["tool_calls_executed"] == []
    assert "Hallucinated tool call" in caplog.text
