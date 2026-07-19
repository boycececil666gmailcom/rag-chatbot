from langchain_core.tools import tool
import src.chatbot_backend.vector_db as db
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

def retrieve_local_documents_raw(query: str) -> tuple[str, list[dict]]:
    """Query the local vector database and return formatted context and raw document metadata."""
    try:
        store = db.get_vector_store()
    except Exception as e:
        return f"Error: Local Vector database is not initialized: {e}", []
    try:
        # Perform hybrid search natively in Qdrant (which uses RRF under the hood)
        docs = store.similarity_search(query, k=5)
        
        if not docs:
            return "No matching local documents found.", []
            
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
        doc_list = []
        for doc in reranked_docs:
            score = doc.metadata.get("relevance_score", 0.0)
            context_list.append(f"[Match Score: {score:.3f}] Content: {doc.page_content}")
            
            # Convert any numpy types to Python types to avoid Pydantic serialization errors
            cleaned_metadata = {}
            for k, v in doc.metadata.items():
                if hasattr(v, "item"):
                    cleaned_metadata[k] = v.item()
                else:
                    cleaned_metadata[k] = v
                    
            doc_list.append({
                "page_content": doc.page_content,
                "metadata": cleaned_metadata
            })
            
        return "\n\n".join(context_list), doc_list
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error querying local documents: {str(e)}", []

@tool
def retrieve_local_documents(query: str) -> str:
    """Retrieve semantically relevant document chunks from the local vector database.
    Use this tool when the query refers to private documentation, internal guidelines,
    project names (like 'Supernova'), or local workspace facts."""
    context_str, _ = retrieve_local_documents_raw(query)
    return context_str
