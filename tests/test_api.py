import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.config import OLLAMA_MODEL

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == OLLAMA_MODEL

@patch("src.main.llm_with_tools")
def test_query_no_search(mock_llm_with_tools):
    # Mock LLM behavior:
    # First invocation -> returns mock answer with no tool calls
    mock_res = MagicMock()
    mock_res.content = "Paris is the capital of France."
    mock_res.tool_calls = []
    
    mock_llm_with_tools.invoke.return_value = mock_res
    
    response = client.post("/query", json={"message": "What is the capital of France?"})
    
    assert response.status_code == 200
    assert response.json()["response"] == "Paris is the capital of France."
    mock_llm_with_tools.invoke.assert_called_once()

@patch("src.main.search_tool")
@patch("src.main.llm_with_tools")
def test_query_with_search(mock_llm_with_tools, mock_search_tool):
    # Mock LLM behavior:
    # First invocation -> returns tool calls to search
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "duckduckgo_search",
        "args": {"query": "Who is the current USA president?"},
        "id": "call_123"
    }]
    
    # Second invocation -> returns search-based answer
    mock_res_answer = MagicMock()
    mock_res_answer.content = "The current US president is Donald John Trump."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    # Mock search results
    mock_search_tool.invoke.return_value = "Donald John Trump assumed office in 2025."
    
    response = client.post("/query", json={"message": "Who is the current USA president?"})
    
    assert response.status_code == 200
    assert response.json()["response"] == "The current US president is Donald John Trump."
    mock_search_tool.invoke.assert_called_once_with({"query": "Who is the current USA president?"})
    assert mock_llm_with_tools.invoke.call_count == 2
