import os
import sys
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.theme_based_rag_backend.config import QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY, GEMINI_EMBED_MODEL

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize embeddings and vector DB lazily
embeddings = None
sparse_embeddings = None
vector_store = None
init_error = None

def get_vector_store():
    global vector_store, init_error, embeddings, sparse_embeddings
    if vector_store is not None:
        return vector_store
    #we need gemini api key to access the embedding model
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if embeddings is None:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBED_MODEL,
                google_api_key=GEMINI_API_KEY
            )
            print(f"Initialized Google Gemini Embeddings Model: {GEMINI_EMBED_MODEL}")
            
        if sparse_embeddings is None:
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            print("Initialized FastEmbed BM25 Sparse Embeddings Model")
            
        if "pytest" in sys.modules or QDRANT_URL == ":memory:":
            print("Running in-memory Qdrant Client for testing...")
            vector_store = QdrantVectorStore.from_documents(
                [],
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                location=":memory:",
                collection_name="local_rag_documents",
                retrieval_mode=RetrievalMode.HYBRID
            )
        elif QDRANT_URL:
            print(f"Connecting to remote Qdrant DB server at {QDRANT_URL}")
            try:
                vector_store = QdrantVectorStore.from_existing_collection(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name="local_rag_documents",
                    embedding=embeddings,
                    sparse_embedding=sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID
                )
            except Exception:
                print("Collection 'local_rag_documents' not found. Creating a new one...")
                vector_store = QdrantVectorStore.from_documents(
                    [],
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name="local_rag_documents",
                    embedding=embeddings,
                    sparse_embedding=sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID
                )
        else:
            raise ValueError("QDRANT_URL environment variable is not configured.")
        init_error = None
        return vector_store
    except Exception as e:
        init_error = str(e)
        print(f"Error initializing vector database or embeddings: {e}")
        raise e

# Initial attempt during import, but don't block start if it fails
try:
    get_vector_store()
except Exception:
    pass

def add_document_text(text: str, metadata: dict = None) -> int:
    """Chunks text and adds documents to Qdrant vector database. Returns chunk count."""
    store = get_vector_store()
    chunks = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata=metadata or {})
        for chunk in chunks
    ]
    store.add_documents(documents)
    return len(documents)
