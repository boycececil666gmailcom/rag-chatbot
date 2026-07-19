import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api_gateway.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Fintech RAG Chatbot API Gateway")

# Read downstream backend endpoint configuration
CHATBOT_BACKEND_URL = os.getenv("CHATBOT_BACKEND_URL", "http://localhost:8000")
GATEWAY_HOST = os.getenv("HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("PORT", "8080"))

# Configure CORS origins from environment variable, default to allowing all (*)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# If "*" is in allowed origins, we must disable allow_credentials to prevent FastAPI startup failure.
allow_credentials = True
if "*" in ALLOWED_ORIGINS:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize an async HTTP client for proxy routing
async_client = httpx.AsyncClient(timeout=60.0)

@app.post("/ingest", response_model=IngestResponse)
async def route_ingest(request: IngestRequest):
    """Proxies ingestion requests downstream to the core RAG backend."""
    target_url = f"{CHATBOT_BACKEND_URL.rstrip('/')}/ingest"
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [1/2] [{os.path.basename(__file__)}] API Gateway proxying ingestion request to: {target_url}\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    try:
        response = await async_client.post(target_url, json=request.dict())
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Downstream error: {response.text}"
            )
        
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [2/2] [{os.path.basename(__file__)}] API Gateway received success response from backend\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        return IngestResponse(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"Failed connecting to downstream backend at {target_url}: {exc}")
        raise HTTPException(
            status_code=503, 
            detail=f"Downstream service unavailable: {str(exc)}"
        )

@app.post("/query", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Proxies query requests downstream to the core RAG backend."""
    target_url = f"{CHATBOT_BACKEND_URL.rstrip('/')}/query"
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [1/2] [{os.path.basename(__file__)}] API Gateway proxying query request to: {target_url}\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    try:
        response = await async_client.post(target_url, json=request.dict())
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Downstream error: {response.text}"
            )
            
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [2/2] [{os.path.basename(__file__)}] API Gateway received success response from backend\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        return QueryResponse(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"Failed connecting to downstream backend at {target_url}: {exc}")
        raise HTTPException(
            status_code=503, 
            detail=f"Downstream service unavailable: {str(exc)}"
        )

@app.get("/health")
async def health_check():
    """Confirms gateway is running and pings downstream backend to verify full network connection path."""
    backend_status = "unreachable"
    target_url = f"{CHATBOT_BACKEND_URL.rstrip('/')}/health"
    try:
        response = await async_client.get(target_url)
        if response.status_code == 200:
            backend_status = "healthy"
        else:
            backend_status = f"unhealthy (status {response.status_code})"
    except Exception as e:
        logger.warning(f"Health check failed to contact downstream backend: {e}")
        
    return {
        "status": "ok",
        "service": "Fintech RAG Chatbot API Gateway",
        "downstream_backend": {
            "endpoint": CHATBOT_BACKEND_URL,
            "status": backend_status
        }
    }

if __name__ == "__main__":
    uvicorn.run("src.api_gateway.main:app", host=GATEWAY_HOST, port=GATEWAY_PORT, reload=True)
