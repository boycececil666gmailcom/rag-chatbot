import os
import httpx
import pytest

# Read live Gateway URL from environment variables, default to port-forward port 8080
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")

def test_e2e_health():
    """Verify that the live health check endpoint responds correctly."""
    response = httpx.get(f"{GATEWAY_URL}/health", timeout=30.0)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert res["service"] == "Fintech RAG Chatbot API Gateway"
    assert "status" in res["downstream_backend"]

def test_e2e_query_refusal():
    """Verify that an out-of-scope query triggers a direct refusal and no tools are run."""
    response = httpx.post(
        f"{GATEWAY_URL}/query",
        json={"message": "What is the capital of France?", "history": []},
        timeout=30.0
    )
    assert response.status_code == 200
    res = response.json()
    assert "response" in res
    assert "Fintech" in res["response"] or "knowledge base" in res["response"]
    assert isinstance(res["tool_calls_executed"], list)

def test_e2e_ingest_and_query_success():
    """Verify ingestion of a document and its successful hybrid retrieval."""
    # 1. Ingest document containing a unique platform metadata/keyword
    ingest_response = httpx.post(
        f"{GATEWAY_URL}/ingest",
        json={
            "text": "Our Fintech SaaS platform uses a proprietary transaction routing mechanism code-named AegisSec-99.",
            "metadata": {"source": "e2e_hybrid_test"}
        },
        timeout=30.0
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"
    assert ingest_response.json()["chunk_count"] > 0
    
    # 2. Query for the ingested fact using the keyword
    response = httpx.post(
        f"{GATEWAY_URL}/query",
        json={"message": "What is the routing mechanism code name used by the SaaS platform?", "history": []},
        timeout=30.0
    )
    assert response.status_code == 200
    res = response.json()
    assert "response" in res
    # Assert that it retrieved the answer AegisSec-99
    assert "AegisSec-99" in res["response"]
    assert "retrieve_local_documents" in res["tool_calls_executed"]

