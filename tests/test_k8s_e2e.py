import os
import time
import httpx
import pytest

GATEWAY_URL = os.getenv("K8S_GATEWAY_URL", "http://localhost")
HOST_HEADER = os.getenv("K8S_HOST_HEADER", "theme-based-rag-workflow.local")

def test_k8s_real_e2e_flow():
    """Real end-to-end integration test running against the actual services deployed in Kubernetes via Ingress."""
    client = httpx.Client(timeout=60.0, headers={"Host": HOST_HEADER})
    
    # 1. Ingest document via K8s Gateway
    ingest_payload = {
        "text": "The Aurora Project is an experimental quantum computing initiative developed by Zenith Tech for our Fintech SaaS platform.",
        "metadata": {"project": "Aurora"}
    }
    print(f"\nSending ingest request to: {GATEWAY_URL}/ingest (Host: {HOST_HEADER})")
    ingest_response = client.post(f"{GATEWAY_URL}/ingest", json=ingest_payload)
    
    assert ingest_response.status_code == 200, f"Ingestion failed: {ingest_response.text}"
    res_ingest = ingest_response.json()
    assert res_ingest["status"] == "success"
    assert res_ingest["chunk_count"] > 0
    print(f"Ingestion successful! Chunks: {res_ingest['chunk_count']}")
    
    # Give some time for Qdrant index to commit
    time.sleep(2)
    
    # 2. Query chatbot via K8s Gateway (Theme-based query)
    query_payload = {
        "message": "What is the Aurora Project on the Fintech SaaS platform?"
    }
    print(f"Sending query request to: {GATEWAY_URL}/query")
    query_response = client.post(f"{GATEWAY_URL}/query", json=query_payload)
    
    assert query_response.status_code == 200, f"Query failed: {query_response.text}"
    res_query = query_response.json()
    print(f"Response: {res_query}")
    
    # Verify response is grounded and contains the correct project name
    assert "Zenith Tech" in res_query["response"] or "quantum computing" in res_query["response"]
    assert "retrieve_local_documents" in res_query["tool_calls_executed"]
    assert "Aurora Project" in res_query["retrieved_documents"]
    
    # 3. Query chatbot with off-theme query (should trigger refusal node)
    refusal_payload = {
        "message": "How do I make a chocolate cake?"
    }
    print(f"Sending off-theme query request to: {GATEWAY_URL}/query")
    refusal_response = client.post(f"{GATEWAY_URL}/query", json=refusal_payload)
    
    assert refusal_response.status_code == 200
    res_refusal = refusal_response.json()
    print(f"Refusal Response: {res_refusal}")
    
    # Should be categorized as refuse and not execute document retrieval
    assert "retrieve_local_documents" not in res_refusal["tool_calls_executed"]
    # The response should politely decline
    assert any(kw in res_refusal["response"].lower() for kw in ["sorry", "decline", "unable", "cannot", "only assist", "fintech"])
