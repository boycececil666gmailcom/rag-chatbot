import os
import uvicorn
import httpx
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# Import configurations
from src.config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, HOST, PORT

app = FastAPI(title="Ollama Web Search Backend")

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str

# Initialize Ollama model and DuckDuckGo Search tool
init_error = None
llm = None
search_tool = None

def check_and_pull_model():
    """Verify Ollama status and ensure the model is downloaded."""
    try:
        # Check if Ollama is running
        res = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        res.raise_for_status()
        
        # Check if model is downloaded
        models = [m["name"] for m in res.json().get("models", [])]
        if OLLAMA_MODEL not in models and f"{OLLAMA_MODEL}:latest" not in models:
            print(f"Model '{OLLAMA_MODEL}' is missing. Pulling from Ollama...")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            print(f"Model '{OLLAMA_MODEL}' successfully downloaded.")
    except Exception as e:
        print(f"Warning/Error checking Ollama status: {e}")

try:
    check_and_pull_model()
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
    search_tool = DuckDuckGoSearchRun()
except Exception as e:
    init_error = str(e)
    print(f"Error initializing LangChain components: {e}")

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if llm is None or search_tool is None:
        raise HTTPException(
            status_code=500, 
            detail=f"LLM/Search is not initialized. Actual Error: {init_error}. (Ensure Ollama is running and all dependencies are installed)"
        )
    
    try:
        query = request.message
        
        # 1. Classify if query needs real-time search
        classification_prompt = (
            "Instruction: Decide if the following question asks about recent events, current facts, or real-time details (like current president, today's news, current weather, current dates) that require an internet search to be accurate.\n"
            f"Question: {query}\n"
            "Does this require an internet search? Answer with ONLY 'YES' or 'NO' (no other text)."
        )
        classification_res = llm.invoke(classification_prompt).content.strip().upper()
        
        print(f"\n--- Query: '{query}' ---")
        print(f"Classification: '{classification_res}'")
        
        if "YES" in classification_res:
            print("Searching...")
            search_results = search_tool.run(query)
            print(f"Search Results retrieved successfully.")
            
            # 2. Feed search results as context
            prompt = (
                f"Use the search results below to answer the query.\n\n"
                f"Search Results:\n{search_results}\n\n"
                f"Query: {query}"
            )
            response = llm.invoke(prompt)
        else:
            print("Answering directly...")
            response = llm.invoke(query)
            
        print("Response Generated successfully.\n")
        return QueryResponse(response=response.content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL, "search": "DuckDuckGo"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)
