import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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
llm_with_tools = None

try:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
    search_tool = DuckDuckGoSearchRun()
    llm_with_tools = llm.bind_tools([search_tool])
except Exception as e:
    init_error = str(e)
    print(f"Error initializing LangChain components: {e}")

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if llm is None or search_tool is None or llm_with_tools is None:
        raise HTTPException(
            status_code=500, 
            detail=f"LLM/Search is not initialized. Actual Error: {init_error}. (Ensure Ollama is running and all dependencies are installed)"
        )
    
    try:
        query = request.message
        print(f"\n--- Query: '{query}' ---")
        
        messages = [HumanMessage(content=query)]
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            print(f"Tool calls requested: {response.tool_calls}")
            messages.append(response)
            for tool_call in response.tool_calls:
                try:
                    tool_output = search_tool.invoke(tool_call["args"])
                except Exception as tool_err:
                    print(f"Direct tool invoke failed, extracting query value: {tool_err}")
                    q_val = (
                        tool_call["args"].get("query")
                        or tool_call["args"].get("input")
                        or list(tool_call["args"].values())[0]
                        if isinstance(tool_call["args"], dict) and tool_call["args"]
                        else str(tool_call["args"])
                    )
                    tool_output = search_tool.invoke(q_val)
                
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)
            
            print("Invoking model again with search results...")
            final_response = llm_with_tools.invoke(messages)
            response_content = final_response.content
        else:
            print("Answering directly...")
            response_content = response.content
            
        print("Response Generated successfully.\n")
        return QueryResponse(response=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL, "search": "DuckDuckGo"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)
