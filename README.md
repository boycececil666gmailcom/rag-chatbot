# Theme-Based RAG Workflow

A modular, stateless Retrieval-Augmented Generation (RAG) customer service chatbot utilizing the Google Gemini API and Qdrant for document storage.

## Business & Product Flow (Overview)

Below is a simplified view of how information flows through the Theme-Based RAG Workflow system, designed for product managers and operations:

```mermaid
flowchart TD
    %% Styling Node classes
    classDef client fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef router fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef kb fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8;
    classDef reply fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#166534;
    classDef block fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#991b1b;
    classDef process fill:#f9fafb,stroke:#d1d5db,stroke-width:1px,color:#374151;
    
    %% Main customer interaction entry point
    Client(["📱 Client App / Customer"]):::client
    
    Client -->|"1. Submits chat query"| Classifier{"🤖 Classifier Node (Gemini)"}:::router
    
    %% Classifier branches
    Classifier -->|"Theme Query (rag)"| QA["🔍 RAG QA Node"]:::process
    Classifier -->|"General Query (refuse)"| Safeguard["🛡️ Safeguard Node"]:::block
    
    %% Qdrant DB Retrieval
    QA -->|"Retrieve context"| DB[("📚 Knowledge Base (Qdrant)")]:::kb
    DB -->|"Return source text"| QA
    
    %% Critique verification
    QA -->|"2. Draft response"| Critique{"🔎 Critique Node (Groundedness Check)"}:::router
    Safeguard -->|"Polite refusal draft"| Critique
    
    %% Critique outcomes
    Critique -->|"PASS (or Max 3 Attempts)"| VerifiedReply["Verified Answer"]:::reply
    Critique -->|"FAIL (Loop back to refine)"| Classifier
    
    VerifiedReply -->|"3. Returns response"| Client
    
    %% Ingestion background flow
    subgraph Ingestion ["Knowledge Ingestion (Offline Feed)"]
        style Ingestion fill:#f9fafb,stroke:#d1d5db,stroke-width:1px;
        Admin(["Product / Ops Admin"]):::client -->|"Uploads FAQs & Guides"| Split["✂️ Text Splitter"]:::process
        Split -->|"Generate Dense & Sparse Embeddings"| DB
    end
```

---

## Features & API Endpoints

The backend exposes two main HTTP POST endpoints under FastAPI:

- **`POST /ingest`**: Accepts raw text documents, splits them into manageable chunks (using `RecursiveCharacterTextSplitter`), generates dense/sparse embeddings, and stores them in the Qdrant database.
- **`POST /query`**: Accepts user queries and conversation history. An LLM agent routes queries to retrieve platform documentation from Qdrant. If the query does not match the configured theme, direct generation is refused to keep responses strictly grounded.
- **`GET /health`**: Performs liveness checks, confirming connection to the vector store downstream.

---

## Configuration

The application is configured using environment variables (stored locally in a `.env` file).

| Environment Variable | Description | Default Value |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API credentials | *(Required)* |
| `GEMINI_MODEL` | Gemini LLM model for routing and synthesis | `gemini-3.1-flash-lite` |
| `GEMINI_EMBED_MODEL` | Google Generative AI embeddings model | `gemini-embedding-001` |
| `GEMINI_TEMPERATURE` | Generation temperature (0.0 for deterministic RAG answers) | `0.0` |
| `PORT` | FastAPI server port for Chatbot Backend | `8000` |
| `HOST` | FastAPI server bind address | `0.0.0.0` |
| `QDRANT_URL` | URL to access the Qdrant database instance (e.g. `http://localhost:6333` or `:memory:`) | *(Required)* |
| `QDRANT_API_KEY` | Optional API Key if using Qdrant Cloud | `None` |
| `CHATBOT_THEME` | The primary theme boundary for retrieval routing & safeguards | `Fintech SaaS platform` |

---

## Architecture & Logic Flow

Below is a high-level flowchart showing how ingestion and querying are routed through the FastAPI backend:

```mermaid
flowchart TD
    %% Styling classes
    classDef main fill:#f9fafb,stroke:#d1d5db,stroke-width:1px,color:#374151;
    classDef ingest fill:#ecfdf5,stroke:#10b981,stroke-width:1px,color:#065f46;
    classDef lgNode fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e40af;
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef endNode fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8;

    Server[FastAPI Server]:::main
    
    %% Ingest path
    Server -->|POST /ingest| Ingest[Document Ingestion Path]:::ingest
    Ingest --> Split[RecursiveCharacterTextSplitter]:::ingest
    Split --> Embed[Gemini Dense / FastEmbed Sparse]:::ingest
    Embed --> DB[(Qdrant DB)]:::ingest

    %% Query path
    Server -->|POST /query| Query[Query Processing Path]:::main
    
    subgraph LangGraph ["🤖 LangGraph Agent Flow (Highlighted)"]
        style LangGraph fill:#f0f7ff,stroke:#2563eb,stroke-width:3px,stroke-dasharray: 5 5;
        
        Graph[Agent Coordinator]:::lgNode
        Classifier{Classifier Node}:::decision
        QA[RAG QA Node]:::lgNode
        Safeguard[Safeguard Node]:::lgNode
        Critique[Critique Node]:::lgNode
        End([End & Return]):::endNode
        
        Graph --> Classifier
        Classifier -->|rag| QA
        Classifier -->|refuse| Safeguard
        
        QA --> Critique
        Safeguard --> Critique
        
        Critique -->|PASS / Max Attempts| End
        Critique -->|FAIL| Classifier
    end
    
    Query --> Graph
```

### 1. Ingestion Path

The ingestion pipeline splits input text and uploads semantic chunks (with both dense Gemini and sparse BM25 embeddings) to the Qdrant database.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / Ingestion Script
    participant App as FastAPI Server (main.py)
    participant VectorStore as Qdrant DB

    Client->>App: POST /ingest {"text": "...", "metadata": {...}}
    Note over App: Chunks text using<br/>RecursiveCharacterTextSplitter
    
    alt Ingestion Success
        rect rgb(220, 252, 231)
            App->>VectorStore: Add document chunks (Dense + Sparse embeddings)
            VectorStore-->>App: Confirmation
            App-->>Client: Response {"status": "success", "chunk_count": X}
        end
    else Ingestion Failure (Database Offline / Missing Credentials)
        rect rgb(254, 226, 226)
            App->>VectorStore: Connection Error / Missing Key
            App-->>Client: HTTP 500 Internal Server Error
        end
    end
```

### 2. Query Path

When a query is received, the request is dispatched to a stateful LangGraph agent workflow containing dynamic classification, generation nodes, and grounding evaluation:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as FastAPI Server (main.py)
    participant Graph as LangGraph Agent (agent_flow)
    participant Classifier as Classifier Node (Embeddings)
    participant QA as RAG QA Node (Gemini)
    participant Safeguard as Safeguard Node (Gemini)
    participant Critique as Critique Node (Gemini)
    participant VectorStore as Qdrant DB

    User->>App: POST /query {"message": "...", "history": [...]}
    App->>Graph: ainvoke(state)
    
    loop Max 3 attempts
        Graph->>Classifier: Determine category (rag / refuse)
        Classifier-->>Graph: Category result
        
        alt Path A: category is 'rag'
            rect rgb(224, 242, 254)
                Graph->>VectorStore: retrieve_local_documents
                VectorStore-->>Graph: Chunks & FlashRank Reranked Docs
                Graph->>QA: Generate draft response using retrieved docs
                QA-->>Graph: draft_response
            end
        else Path B: category is 'refuse'
            rect rgb(254, 226, 226)
                Graph->>Safeguard: Generate polite refusal response
                Safeguard-->>Graph: draft_response
            end
        end
        
        Graph->>Critique: Evaluate draft_response (groundedness / theme-adherence)
        
        alt Validation Passes (or max attempts reached)
            rect rgb(220, 252, 231)
                Critique-->>Graph: Status: PASS
                Note over Graph: Exit loop
            end
        else Validation Fails
            rect rgb(254, 226, 226)
                Critique-->>Graph: Status: FAIL (Reason details)
                Note over Graph: Increment attempts & loop back to Classifier
            end
        end
    end
    
    Graph-->>App: Final state result
    App-->>User: Response {"response": "...", "tool_calls_executed": [...]}
```

---

## Local Development Setup

To run the chatbot and api gateway services locally on your machine, follow these steps:

### Prerequisites

- Python 3.10 or higher
- Docker Desktop (required to run Qdrant database locally)

### 1. Environment Setup

Configure your Python environment and dependencies:
```bash
./setup_env.sh
```
Or manually:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
QDRANT_URL=http://localhost:6333
CHATBOT_THEME=Fintech SaaS platform
```

### 3. Run Qdrant Database

Start a local instance of Qdrant Vector DB:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

### 4. Start Chatbot Backend

Run the backend API (FastAPI) on port 8000:
```bash
# Ensure virtualenv is active
python -m uvicorn src.theme_based_rag_backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start API Gateway

Run the gateway API (FastAPI proxy) on port 8080:
```bash
# Ensure virtualenv is active
python -m uvicorn src.theme_based_rag_gateway.main:app --host 0.0.0.0 --port 8080 --reload
```