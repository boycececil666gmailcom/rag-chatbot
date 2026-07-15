import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.config import OLLAMA_MODEL
from langchain_core.documents import Document

client = TestClient(app)

def test_health_check():
    with patch("src.vector_db.vector_store") as mock_vector:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model"] == OLLAMA_MODEL

@patch("src.vector_db.vector_store")
def test_ingest_endpoint(mock_vector_store):
    # Mock Chroma add_documents
    mock_vector_store.add_documents.return_value = None
    
    response = client.post(
        "/ingest",
        json={
            "text": "Hello World. This is project Supernova 9.",
            "metadata": {"source": "test"}
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunk_count"] > 0
    mock_vector_store.add_documents.assert_called_once()

def test_flashrank_rerank():
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    query = "project Supernova"
    doc1 = Document(page_content="This is project Supernova 9.", metadata={})
    doc2 = Document(page_content="This is a completely unrelated file.", metadata={})
    
    compressor = FlashrankRerank(top_n=2)
    reranked = compressor.compress_documents([doc2, doc1], query)
    assert len(reranked) == 2
    # doc1 should rank first
    assert reranked[0].page_content == "This is project Supernova 9."
    assert "relevance_score" in reranked[0].metadata

def test_reciprocal_rank_fusion():
    from src.rrf import reciprocal_rank_fusion
    
    doc1 = Document(page_content="Supernova project is a quantum framework.", metadata={})
    doc2 = Document(page_content="RAG orchestration details.", metadata={})
    doc3 = Document(page_content="Completely unrelated document content.", metadata={})
    
    # Dense results (dist 0.1 is best, doc1 first, doc2 second)
    dense_results = [
        (doc1, 0.1),
        (doc2, 0.5)
    ]
    # Sparse results (score 10.0 is best, doc2 first, doc1 second, doc3 third)
    sparse_results = [
        (doc2, 10.0),
        (doc1, 5.0),
        (doc3, 1.0)
    ]
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=2)
    assert len(fused) == 2
    assert doc3 not in fused

@patch("src.main.llm_with_tools")

def test_query_no_tools(mock_llm_with_tools):
    mock_res = MagicMock()
    mock_res.content = "Paris is the capital of France."
    mock_res.tool_calls = []
    mock_llm_with_tools.invoke.return_value = mock_res
    
    response = client.post("/query", json={"message": "What is the capital of France?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "Paris is the capital of France."
    assert response.json()["tool_calls_executed"] == []
    assert response.json()["fallback_triggered"] == False

@patch("src.vector_db.vector_store")
@patch("src.main.llm_with_tools")
def test_query_local_vector_retrieval(mock_llm_with_tools, mock_vector_store):
    # First invoke returns tool call to local db search
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "retrieve_local_documents",
        "args": {"query": "Supernova 9"},
        "id": "call_1"
    }]
    
    # Second invoke returns final compiled response
    mock_res_answer = MagicMock()
    mock_res_answer.content = "Supernova 9 is our internal pipeline."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    # Mock Vector search results
    doc = Document(page_content="This is project Supernova 9.", metadata={})
    mock_vector_store.similarity_search_with_score.return_value = [(doc, 0.1)]
    mock_vector_store.get.return_value = {
        "documents": ["This is project Supernova 9."],
        "metadatas": [{}]
    }
    
    response = client.post("/query", json={"message": "What is Supernova 9?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "Supernova 9 is our internal pipeline."
    assert "retrieve_local_documents" in response.json()["tool_calls_executed"]
    assert response.json()["fallback_triggered"] == False

@patch("src.main.llm_with_tools")
def test_query_tool_hallucination_safeguard(mock_llm_with_tools):
    # Simulate a hallucinated tool call name 'non_existent_tool'
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "non_existent_tool",
        "args": {"query": "something"},
        "id": "call_fake"
    }]
    
    # Since tool call is invalid, it triggers safeguard fallback and re-invokes LLM
    mock_res_answer = MagicMock()
    mock_res_answer.content = "This is a direct response output."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    response = client.post("/query", json={"message": "Can you run this?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "This is a direct response output."
    assert response.json()["tool_calls_executed"] == []
    assert response.json()["fallback_triggered"] == True

def test_bm25_retriever():
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    doc1 = Document(page_content="The quick brown fox jumps over the lazy dog", metadata={})
    doc2 = Document(page_content="Python is an amazing programming language", metadata={})
    doc3 = Document(page_content="Web search and database retrieval framework", metadata={})
    
    retriever = BM25Retriever.from_documents([doc1, doc2, doc3])
    retriever.k = 1
    
    # Query with exact matching words in doc2
    results = retriever.invoke("programming language")
    assert len(results) == 1
    assert results[0].page_content == "Python is an amazing programming language"

def test_reciprocal_rank_fusion():
    from src.rrf import reciprocal_rank_fusion
    from langchain_core.documents import Document
    
    doc1 = Document(page_content="Doc 1 content", metadata={})
    doc2 = Document(page_content="Doc 2 content", metadata={})
    doc3 = Document(page_content="Doc 3 content", metadata={})
    
    # Mock semantic results: doc1 at rank 1, doc2 at rank 2
    semantic_results = [(doc1, 0.1), (doc2, 0.2)]
    # Mock BM25 results: doc3 at rank 1, doc1 at rank 2
    bm25_results = [(doc3, 10.0), (doc1, 8.0)]
    
    fused = reciprocal_rank_fusion(semantic_results, bm25_results, k=60, top_n=3)
    
    # Doc 1 should be the top recommendation since it appears in both lists
    assert len(fused) == 3
    assert fused[0].page_content == "Doc 1 content"

@patch("src.vector_db.vector_store")
def test_retrieve_local_documents_integration_pipeline(mock_vector_store):
    from src.tools import retrieve_local_documents
    from langchain_core.documents import Document
    
    # 1. Define document candidates
    doc_a = Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={})
    doc_b = Document(page_content="Aegis project is an encryption scheme code-named Aegis.", metadata={})
    doc_c = Document(page_content="Unrelated document content about solar systems.", metadata={})
    
    # 2. Mock Vector Store get() to return all documents for BM25
    mock_vector_store.get.return_value = {
        "documents": [doc_a.page_content, doc_b.page_content, doc_c.page_content],
        "metadatas": [{}, {}, {}]
    }
    
    # 3. Mock dense vector search to return a poor candidate (doc_c) first and relevant candidate (doc_b) second
    # doc_c gets distance 0.1 (high similarity), doc_b gets distance 0.8 (low similarity)
    mock_vector_store.similarity_search_with_score.return_value = [
        (doc_c, 0.1),
        (doc_b, 0.8)
    ]
    
    # 4. Invoke local retrieval
    # Query has exact keywords for doc_b.
    # The sparse search (BM25) will score doc_b very high, while dense search scored doc_c high.
    # RRF fuses them, and FlashRank reranks doc_b to the top.
    result = retrieve_local_documents.invoke("encryption scheme")
    
    # 5. Assertions
    # The top returned context should contain the Aegis document chunk (doc_b)
    # due to BM25 keyword matching, RRF fusion, and FlashRank cross-encoder reranking.
    assert "Aegis" in result
    assert "encryption scheme" in result
    # doc_b should rank higher than doc_c in the final reranked context
    assert result.index("Aegis") < result.index("Unrelated")
