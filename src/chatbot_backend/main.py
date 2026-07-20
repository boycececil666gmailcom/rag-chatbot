import os
import logging
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

from src.chatbot_backend.config import HOST, PORT, GEMINI_MODEL
import src.chatbot_backend.vector_db as db
from src.chatbot_backend.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from src.chatbot_backend.agent_flow import agent_graph

app = FastAPI(title="AI RAG Search Robot Backend")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    try:
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [1/1] [{os.path.basename(__file__)}] Forwarding ingestion request to Vector Store\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        chunk_count = db.add_document_text(request.text, request.metadata)
        return IngestResponse(status="success", chunk_count=chunk_count)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    try:
        # Map input payload to LangGraph state
        inputs = {
            "message": request.message,
            "history": [{"role": msg.role, "content": msg.content} for msg in request.history],
            "category": "refuse",
            "retrieved_documents": None,
            "draft_response": "",
            "critique_feedback": None,
            "attempts": 0
        }
        
        # Execute workflow graph asynchronously
        result = await agent_graph.ainvoke(inputs)
        
        tool_calls_executed = []
        if result.get("retrieved_documents"):
            tool_calls_executed.append("retrieve_local_documents")
            
        return QueryResponse(
            response=result.get("draft_response", ""),
            tool_calls_executed=tool_calls_executed,
            retrieved_documents=result.get("retrieved_documents")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")


@app.get("/health")
async def health_check():
    try:
        db.get_vector_store()
        vector_ok = "ok"
    except Exception as e:
        logger.error(f"Health check failed to initialize vector store: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Vector store initialization failed: {str(e)}"
        )
        
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "platform": "AI RAG Search Robot",
        "vector_store": vector_ok
    }

if __name__ == "__main__":
    uvicorn.run("src.chatbot_backend.main:app", host=HOST, port=PORT, reload=True)
