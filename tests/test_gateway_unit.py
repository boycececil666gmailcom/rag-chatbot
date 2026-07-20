import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import httpx
from src.theme_based_rag_gateway.main import app, RAG_BACKEND_URL

client = TestClient(app)

@patch("src.theme_based_rag_gateway.main.async_client")
def test_health_endpoint_backend_healthy(mock_async_client):
    """Test that the gateway /health endpoint returns a healthy status when the downstream RAG backend is healthy (HTTP 200)."""
    mock_resp = httpx.Response(200, json={"status": "ok"})
    mock_async_client.get = AsyncMock(return_value=mock_resp)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["downstream_backend"]["status"] == "healthy"
    mock_async_client.get.assert_called_once_with(f"{RAG_BACKEND_URL.rstrip('/')}/health")

@patch("src.theme_based_rag_gateway.main.async_client")
def test_health_endpoint_backend_unhealthy(mock_async_client):
    """Test that the gateway /health endpoint returns 'unhealthy (status 500)' when the downstream RAG backend is unhealthy (HTTP 500)."""
    mock_resp = httpx.Response(500, text="Internal Error")
    mock_async_client.get = AsyncMock(return_value=mock_resp)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["downstream_backend"]["status"] == "unhealthy (status 500)"

@patch("src.theme_based_rag_gateway.main.async_client")
def test_health_endpoint_backend_exception(mock_async_client):
    """Test that the gateway /health endpoint returns 'unreachable' when a connection error occurs with the downstream RAG backend."""
    mock_async_client.get = AsyncMock(side_effect=httpx.RequestError("Connection refused"))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["downstream_backend"]["status"] == "unreachable"

@patch("src.theme_based_rag_gateway.main.async_client")
def test_ingest_endpoint_success(mock_async_client):
    """Test that the document ingestion (/ingest) endpoint succeeds when the downstream backend processes it successfully."""
    mock_resp = httpx.Response(200, json={"status": "success", "chunk_count": 5})
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    payload = {"text": "Document text", "metadata": {"key": "value"}}
    response = client.post("/ingest", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "chunk_count": 5}
    mock_async_client.post.assert_called_once()

@patch("src.theme_based_rag_gateway.main.async_client")
def test_ingest_endpoint_downstream_error(mock_async_client):
    """Test that the document ingestion (/ingest) endpoint correctly propagates errors when the downstream backend returns an HTTP 400 error."""
    mock_resp = httpx.Response(400, text="Bad Request downstream")
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    payload = {"text": "Document text", "metadata": {"key": "value"}}
    response = client.post("/ingest", json=payload)
    
    assert response.status_code == 400
    assert "Downstream error" in response.json()["detail"]

@patch("src.theme_based_rag_gateway.main.async_client")
def test_ingest_endpoint_connection_error(mock_async_client):
    """Test that the document ingestion (/ingest) endpoint returns an HTTP 503 error when a connection error occurs with the downstream backend."""
    mock_async_client.post = AsyncMock(side_effect=httpx.RequestError("Gateway timeout"))

    payload = {"text": "Document text", "metadata": {"key": "value"}}
    response = client.post("/ingest", json=payload)
    
    assert response.status_code == 503
    assert "Downstream service unavailable" in response.json()["detail"]

@patch("src.theme_based_rag_gateway.main.async_client")
def test_query_endpoint_success(mock_async_client):
    """Test that the query (/query) endpoint successfully retrieves documents and returns the generated answer from the downstream backend."""
    mock_resp = httpx.Response(200, json={
        "response": "Final Answer",
        "tool_calls_executed": ["retrieve_local_documents"],
        "retrieved_documents": "Retrieved chunks"
    })
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    payload = {
        "message": "User query",
        "history": [{"role": "user", "content": "hello"}]
    }
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {
        "response": "Final Answer",
        "tool_calls_executed": ["retrieve_local_documents"],
        "retrieved_documents": "Retrieved chunks"
    }

@patch("src.theme_based_rag_gateway.main.async_client")
def test_query_endpoint_downstream_error(mock_async_client):
    """Test that the query (/query) endpoint correctly propagates errors when the downstream backend returns an HTTP 500 error."""
    mock_resp = httpx.Response(500, text="Internal Server Error")
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    payload = {"message": "User query"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 500
    assert "Downstream error" in response.json()["detail"]
