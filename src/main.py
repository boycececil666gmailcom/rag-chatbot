import os
import logging
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Import configuration
from src.config import HOST, PORT, GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

# Import modular components
import src.vector_db as db
from src.tools import retrieve_local_documents
from src.models import MessageSchema, QueryRequest, QueryResponse, IngestRequest, IngestResponse

app = FastAPI(title="Fintech RAG Chatbot")

# Initialize LLM & Tool binding
llm = None
llm_with_tools = None

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=GEMINI_TEMPERATURE
    )
    print(f"Initialized Google Gemini Chat Model: {GEMINI_MODEL}")
    llm_with_tools = llm.bind_tools([retrieve_local_documents])
except Exception as e:
    print(f"Error initializing LLM: {e}")

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
    if llm is None or llm_with_tools is None:
        raise HTTPException(
            status_code=500, 
            detail="LLM component is not initialized."
        )
    
    tool_calls_executed = []
    
    try:
        # Step 1/4: Payload mapping
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [1/4] [{os.path.basename(__file__)}] Parsing request and history payload\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        messages = [
            SystemMessage(content=(
                "You are a customer service assistant for a Fintech RAG Chatbot.\n"
                "You have access to a local vector database containing platform documentation via the tool 'retrieve_local_documents'.\n"
                "CRITICAL RULES:\n"
                "1. If the user's query is unrelated to the Fintech SaaS platform (e.g. general knowledge, math, other countries, capitals like France/Paris, etc.), you MUST NOT call any tools. You must answer directly and politely refuse to respond, stating that you can only help with inquiries related to the Fintech RAG Chatbot.\n"
                "2. Only call 'retrieve_local_documents' if the query is specifically about bank accounts, SaaS platform usage, account creation, transfers, fees, features, security, internal guidelines, or specific workspace facts."
            ))
        ]
        
        # Hydrate stateless message history
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
                
        # Append current user query
        messages.append(HumanMessage(content=request.message))
        
        # Step 2/4: Agent Routing Decision
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [2/4] [{os.path.basename(__file__)}] Invoking LLM to determine routing paths\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        response = llm_with_tools.invoke(messages)
        
        # Step 3/4: Tool Execution & Safeguards
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [3/4] [{os.path.basename(__file__)}] Executing routing tools with safeguards\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Safeguard 1: Hallucinated / Invalid Tool Name check
                if tool_name not in ["retrieve_local_documents"]:
                    logger.warning(f"Hallucinated tool call '{tool_name}' detected. Triggering safeguard fallback.")
                    continue
                    
                tool_calls_executed.append(tool_name)
                
                # Safeguard 2: Parameter parsing safety
                q_val = (
                    tool_args.get("query")
                    or tool_args.get("input")
                    or list(tool_args.values())[0]
                    if isinstance(tool_args, dict) and tool_args
                    else str(tool_args)
                )
                
                # Run actual tool
                try:
                    if tool_name == "retrieve_local_documents":
                        tool_output = retrieve_local_documents.invoke(q_val)
                except Exception as tool_err:
                    logger.error(f"Tool execution failed: {tool_err}. Triggering direct fallback.")
                    tool_output = f"Error: Failed execution context fallback: {str(tool_err)}"
                
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)
                
            # Step 4/4: Final Synthesis
            print(f"\n\033[1;96m========================================================\033[0m")
            print(f"\033[1;92m>>> [4/4] [{os.path.basename(__file__)}] Synthesizing final answer with tools context\033[0m")
            print(f"\033[1;96m========================================================\033[0m\n")
            
            # Refusal Safeguard: Append a system reminder if the database search returned no results
            if any("No matching local documents found" in str(msg.content) for msg in messages if isinstance(msg, ToolMessage)):
                messages.append(SystemMessage(content=(
                    "Refusal Safeguard: If the retrieved database context is empty and the user query is unrelated "
                    "to the Fintech SaaS platform, you must refuse to answer. Do not use your pre-trained knowledge "
                    "to answer general knowledge questions. Politely state that you can only help with inquiries "
                    "related to the Fintech RAG Chatbot."
                )))
                
            final_response = llm_with_tools.invoke(messages)
            response_content = final_response.content
            if isinstance(response_content, list):
                response_content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response_content
                )
        else:
            # Step 4/4 (Direct Path Refused)
            print(f"\n\033[1;96m========================================================\033[0m")
            print(f"\033[1;92m>>> [4/4] [{os.path.basename(__file__)}] Direct response path refused\033[0m")
            print(f"\033[1;96m========================================================\033[0m\n")
            response_content = "I'm sorry, Chatbot has decided that the query can't be solved by using the hosted knowledge base. It's advised to used general-purpose agent to solve your problem."
            
        print("Query execution completed successfully.\n")
        return QueryResponse(
            response=response_content,
            tool_calls_executed=tool_calls_executed
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/health")
async def health_check():
    vector_ok = "ok" if db.vector_store is not None else "failed"
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "platform": "Fintech RAG Chatbot",
        "vector_store": vector_ok
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)
