import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from src.theme_based_rag_backend.main import app as backend_app
from src.theme_based_rag_gateway.main import app as gateway_app
from src.theme_based_rag_gateway import main as gateway_module

@pytest.fixture(autouse=True)
def setup_gateway_routing():
    """Fixture to route gateway asynchronous HTTP requests directly to the backend FastAPI ASGI application in-memory."""
    # Route gateway request client in-memory directly to backend ASGI app
    gateway_module.async_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend_app), base_url="http://localhost:8000")
    yield

@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_full_e2e_flow(mock_llm):
    """Test the full end-to-end flow from document ingestion to querying via the API Gateway."""
    gateway_client = TestClient(gateway_app)
    
    # 1. Ingest document via API Gateway
    ingest_payload = {
        "text": "Supernova project is a next generation security scanning system developed for Fintech platforms.",
        "metadata": {"project": "Supernova"}
    }
    ingest_response = gateway_client.post("/ingest", json=ingest_payload)
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"
    assert ingest_response.json()["chunk_count"] > 0

    # 2. Query chatbot via API Gateway
    # Mock LLM nodes sequentially: RAG QA -> Critique (Classifier is now vector-similarity based, no LLM call)
    mock_resp_qa = MagicMock(content='The Supernova project is a next generation security scanning system.')
    mock_resp_crit = MagicMock(content='{"status": "PASS"}')
    
    mock_llm.invoke.side_effect = [mock_resp_qa, mock_resp_crit]

    query_payload = {
        "message": "What is the Supernova project?"
    }
    query_response = gateway_client.post("/query", json=query_payload)
    
    assert query_response.status_code == 200
    res_json = query_response.json()
    assert "Supernova project" in res_json["response"]
    assert "retrieve_local_documents" in res_json["tool_calls_executed"]
    assert "Supernova project is a next generation" in res_json["retrieved_documents"]
