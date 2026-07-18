from langchain_core.tools import tool
import src.chatbot_backend.vector_db as db
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

@tool
def retrieve_local_documents(query: str) -> str:
    """Retrieve semantically relevant document chunks from the local vector database.
    Use this tool when the query refers to private documentation, internal guidelines,
    project names (like 'Supernova'), or local workspace facts."""
    try:
        store = db.get_vector_store()
    except Exception as e:
        return f"Error: Local Vector database is not initialized: {e}"
    try:
        # Perform hybrid search natively in Qdrant (which uses RRF under the hood)
        docs = store.similarity_search(query, k=5)
        
        if not docs:
            return "No matching local documents found."
            
        # Apply FlashRank Cross-Encoder reranker using LangChain's FlashrankRerank
        try:
            compressor = FlashrankRerank(top_n=2)
            reranked_docs = compressor.compress_documents(docs, query)
        except Exception as rerank_err:
            import logging
            logging.getLogger(__name__).warning(
                f"FlashRank reranking failed, falling back to database rankings: {rerank_err}"
            )
            reranked_docs = docs[:2]
        
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
