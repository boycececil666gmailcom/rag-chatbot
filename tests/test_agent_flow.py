import pytest
from unittest.mock import patch, MagicMock
from src.theme_based_rag_backend.agent_flow import (
    AgentState,
    classifier_node,
    rag_qa_node,
    safeguard_node,
    critique_node,
    route_by_category,
    route_after_critique,
    agent_graph
)

# Test classifier_node
@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_classifier_node_rag(mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"category": "rag"}'
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "Tell me about Fintech SaaS platform rules",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = classifier_node(state)
    assert result == {"category": "rag"}
    mock_llm.invoke.assert_called_once()

@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_classifier_node_refuse(mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"category": "refuse"}'
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "How to make a cake?",
        "history": [],
        "category": "rag",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = classifier_node(state)
    assert result == {"category": "refuse"}

@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_classifier_node_fallback(mock_llm):
    mock_response = MagicMock()
    mock_response.content = "This belongs to the RAG category."
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "Help me with private documentation",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = classifier_node(state)
    assert result == {"category": "rag"}

# Test safeguard_node
@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_safeguard_node(mock_llm):
    mock_response = MagicMock()
    mock_response.content = "I can only assist with Fintech SaaS platform questions."
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "General query",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = safeguard_node(state)
    assert result == {"draft_response": "I can only assist with Fintech SaaS platform questions."}

# Test rag_qa_node
@patch("src.theme_based_rag_backend.agent_flow.retrieve_local_documents")
@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_rag_qa_node_with_retrieval(mock_llm, mock_retrieve):
    mock_retrieve.invoke.return_value = "Retrieved doc context"
    mock_response = MagicMock()
    mock_response.content = "Synthesized response"
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "query message",
        "history": [],
        "category": "rag",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = rag_qa_node(state)
    
    assert result["draft_response"] == "Synthesized response"
    assert result["retrieved_documents"] == "Retrieved doc context"
    mock_retrieve.invoke.assert_called_once_with("query message")

@patch("src.theme_based_rag_backend.agent_flow.retrieve_local_documents")
@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_rag_qa_node_already_retrieved(mock_llm, mock_retrieve):
    mock_response = MagicMock()
    mock_response.content = "Synthesized response"
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "query message",
        "history": [],
        "category": "rag",
        "retrieved_documents": "Existing documents",
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    result = rag_qa_node(state)
    
    assert result["draft_response"] == "Synthesized response"
    assert result["retrieved_documents"] == "Existing documents"
    mock_retrieve.invoke.assert_not_called()

# Test critique_node
@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_critique_node_refuse(mock_llm):
    state: AgentState = {
        "message": "query message",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "draft_response": "Refused response",
        "critique_feedback": None,
        "attempts": 0
    }
    result = critique_node(state)
    assert result == {"critique_feedback": "PASS"}
    mock_llm.invoke.assert_not_called()

@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_critique_node_pass(mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"status": "PASS"}'
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "query message",
        "history": [],
        "category": "rag",
        "retrieved_documents": "Context",
        "draft_response": "Response",
        "critique_feedback": None,
        "attempts": 1
    }
    result = critique_node(state)
    assert result == {"critique_feedback": "PASS"}

@patch("src.theme_based_rag_backend.agent_flow.llm")
def test_critique_node_fail(mock_llm):
    mock_response = MagicMock()
    mock_response.content = '{"status": "FAIL", "reason": "Hallucination of fact X"}'
    mock_llm.invoke.return_value = mock_response

    state: AgentState = {
        "message": "query message",
        "history": [],
        "category": "rag",
        "retrieved_documents": "Context",
        "draft_response": "Response",
        "critique_feedback": None,
        "attempts": 1
    }
    result = critique_node(state)
    assert result == {"critique_feedback": "Hallucination of fact X", "attempts": 2}

# Test route_by_category
def test_route_by_category():
    state: AgentState = {"category": "rag"}
    assert route_by_category(state) == "rag"
    
    state = {"category": "refuse"}
    assert route_by_category(state) == "refuse"

# Test route_after_critique
def test_route_after_critique():
    state: AgentState = {"critique_feedback": "PASS", "attempts": 0}
    assert route_after_critique(state) == "approved"
    
    state = {"critique_feedback": "Hallucination detected", "attempts": 1}
    assert route_after_critique(state) == "rejected"
    
    state = {"critique_feedback": "Hallucination detected", "attempts": 3}
    assert route_after_critique(state) == "approved"

# Test compiled graph end-to-end
@patch("src.theme_based_rag_backend.agent_flow.retrieve_local_documents")
@patch("src.theme_based_rag_backend.agent_flow.llm")
@pytest.mark.asyncio
async def test_agent_graph_e2e(mock_llm, mock_retrieve):
    mock_retrieve.invoke.return_value = "Retrieved documents"
    
    # Mock LLM calls: Classifier -> RAG QA -> Critique
    mock_resp_class = MagicMock(content='{"category": "rag"}')
    mock_resp_qa = MagicMock(content='Draft Response text')
    mock_resp_crit = MagicMock(content='{"status": "PASS"}')
    
    mock_llm.invoke.side_effect = [mock_resp_class, mock_resp_qa, mock_resp_crit]
    
    inputs = {
        "message": "User question",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    
    result = await agent_graph.ainvoke(inputs)
    assert result["category"] == "rag"
    assert result["draft_response"] == "Draft Response text"
    assert result["critique_feedback"] == "PASS"
