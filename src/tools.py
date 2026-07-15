from typing import List
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
import src.vector_db as db
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from src.rrf import reciprocal_rank_fusion

# Setup search tool
search_tool = DuckDuckGoSearchRun()

def get_all_documents() -> List[Document]:
    """Helper to fetch all documents stored in the Chroma vector database."""
    if db.vector_store is None:
        return []
    results = db.vector_store.get(include=["documents", "metadatas"])
    docs = []
    if results and "documents" in results:
        texts = results["documents"]
        metas = results["metadatas"] or [{}] * len(texts)
        for text, meta in zip(texts, metas):
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs

@tool
def retrieve_local_documents(query: str) -> str:
    """Retrieve semantically relevant document chunks from the local vector database.
    Use this tool when the query refers to private documentation, internal guidelines,
    project names (like 'Supernova'), or local workspace facts."""
    if db.vector_store is None:
        return "Error: Local Vector database is not initialized."
    try:
        # 1. Retrieve dense candidates (top 10)
        dense_results = db.vector_store.similarity_search_with_score(query, k=10)
        
        # 2. Get all documents from vector store to build BM25 index
        all_documents = get_all_documents()
        
        # 3. Perform BM25 sparse search (top 10)
        bm25_results = []
        if all_documents:
            retriever = BM25Retriever.from_documents(all_documents)
            retriever.k = 10
            bm25_docs = retriever.invoke(query)
            bm25_results = [(doc, float(len(bm25_docs) - i)) for i, doc in enumerate(bm25_docs)]
            
        if not dense_results and not bm25_results:
            return "No matching local documents found."
            
        # 4. Fuse using Reciprocal Rank Fusion (RRF) from src.rrf
        fused_docs = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=5)
        
        # 5. Apply FlashRank Cross-Encoder reranker using LangChain's FlashrankRerank
        compressor = FlashrankRerank(top_n=2)
        reranked_docs = compressor.compress_documents(fused_docs, query)
        
        # Format top chunks as output context
        context_list = []
        for doc in reranked_docs:
            score = doc.metadata.get("relevance_score", 0.0)
            context_list.append(f"[Match Score: {score:.3f}] Content: {doc.page_content}")
            
        return "\n\n".join(context_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error querying local documents: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the public internet for current events, today's weather, real-time news,
    active figures, or public facts."""
    try:
        return search_tool.invoke(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"
