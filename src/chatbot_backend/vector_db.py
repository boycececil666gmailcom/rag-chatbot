import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.chatbot_backend.config import CHROMA_PERSIST_DIR, GEMINI_API_KEY, GEMINI_EMBED_MODEL, CHROMA_SERVER_HOST, CHROMA_SERVER_PORT

# Ensure database directory exists if running locally
if not CHROMA_SERVER_HOST:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize embeddings and vector DB lazily
embeddings = None
vector_store = None
init_error = None

def get_vector_store():
    global vector_store, init_error, embeddings
    if vector_store is not None:
        return vector_store
        
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
            
        if CHROMA_SERVER_HOST:
            import chromadb
            port = int(CHROMA_SERVER_PORT) if CHROMA_SERVER_PORT else 8000
            print(f"Connecting to remote self-hosted Chroma DB server at http://{CHROMA_SERVER_HOST}:{port}")
            client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=port)
            vector_store = Chroma(
                client=client,
                collection_name="local_rag_documents",
                embedding_function=embeddings
            )
        else:
            print(f"Initializing local Chroma DB with storage path: {CHROMA_PERSIST_DIR}")
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="local_rag_documents"
            )
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
    """Chunks text and adds documents to Chroma vector database. Returns chunk count."""
    store = get_vector_store()
    chunks = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata=metadata or {})
        for chunk in chunks
    ]
    store.add_documents(documents)
    return len(documents)
