import pytest
import urllib.request
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def is_ollama_running():
    try:
        # Default Ollama local address. ChatOllama queries this endpoint.
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5)
        return True
    except Exception:
        return False

# Dynamically skip E2E tests if Ollama is not active locally (e.g. in GitHub Actions)
pytestmark = pytest.mark.skipif(
    not is_ollama_running(),
    reason="Ollama server is not running/accessible on http://localhost:11434"
)

def test_e2e_query_no_search():
    """Verify factual query that does not require search is resolved directly."""
    response = client.post("/query", json={"message": "What is the capital of France?"})
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert "Paris" in res_data["response"]

def test_e2e_query_with_search():
    """Verify query that requires search is successfully completed."""
    response = client.post("/query", json={"message": "Who won the most recent Super Bowl?"})
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert len(res_data["response"]) > 0
