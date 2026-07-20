import pytest
import logging
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from src.chatbot_backend.main import app
from src.chatbot_backend.config import GEMINI_MODEL

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint returns success state."""
    with patch("src.chatbot_backend.vector_db.vector_store") as mock_vector:
        response = client.get("/health")
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "ok"
        assert res["model"] == GEMINI_MODEL
        assert res["platform"] == "AI RAG Search Robot"

@patch("src.chatbot_backend.vector_db.vector_store")
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

@patch("src.chatbot_backend.main.agent_graph.ainvoke")
def test_query_endpoint_refusal(mock_ainvoke):
    """Verify that out-of-theme queries trigger refusal pathway response."""
    mock_ainvoke.return_value = {
        "draft_response": "I can only help with inquiries related to the configured theme.",
        "category": "refuse",
        "retrieved_documents": None,
        "attempts": 1
    }
    
    response = client.post(
        "/query",
        json={"message": "What is the capital of France?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "only help" in res["response"].lower() or "configured theme" in res["response"].lower()
    assert res["tool_calls_executed"] == []
    mock_ainvoke.assert_called_once()

@patch("src.chatbot_backend.main.agent_graph.ainvoke")
def test_query_endpoint_retrieval(mock_ainvoke):
    """Verify retrieval-based queries return generated response with tool outputs."""
    mock_ainvoke.return_value = {
        "draft_response": "The wire transfer limit is $10,000.",
        "category": "rag",
        "retrieved_documents": "Wire transfer limits are set to $10,000.",
        "attempts": 1
    }
    
    response = client.post(
        "/query",
        json={"message": "What is the wire transfer limit?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "10,000" in res["response"]
    assert "retrieve_local_documents" in res["tool_calls_executed"]
    assert res["retrieved_documents"] is not None
    assert "limits are set" in res["retrieved_documents"]
    mock_ainvoke.assert_called_once()
