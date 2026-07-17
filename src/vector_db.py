import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHROMA_PERSIST_DIR, GEMINI_API_KEY, GEMINI_EMBED_MODEL

# Ensure database directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize embeddings and vector DB
embeddings = None
vector_store = None
init_error = None

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GEMINI_API_KEY
    )
    print(f"Initialized Google Gemini Embeddings Model: {GEMINI_EMBED_MODEL}")
        
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="local_rag_documents"
    )
except Exception as e:
    init_error = str(e)
    print(f"Error initializing vector database or embeddings: {e}")

def add_document_text(text: str, metadata: dict = None) -> int:
    """Chunks text and adds documents to Chroma vector database. Returns chunk count."""
    if vector_store is None:
        raise ValueError(f"Vector Store is not initialized. Error: {init_error}")
    
    chunks = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata=metadata or {})
        for chunk in chunks
    ]
    vector_store.add_documents(documents)
    return len(documents)
