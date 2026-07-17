from typing import List, Dict, Tuple
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int = 60,
    top_n: int = 5
) -> List[Document]:
    """
    Applies Reciprocal Rank Fusion (RRF) on dense and sparse retrieval results.
    dense_results: List of (Document, score) where score is distance (smaller is better).
    sparse_results: List of (Document, score) where score is BM25 score (larger is better).
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # 1. Score dense results (sorted ascending by distance, so index 0 has rank 1)
    sorted_dense = sorted(dense_results, key=lambda x: x[1])
    for rank, (doc, _) in enumerate(sorted_dense, start=1):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
        
    # 2. Score sparse/BM25 results (sorted descending by score, so index 0 has rank 1)
    sorted_sparse = sorted(sparse_results, key=lambda x: x[1], reverse=True)
    for rank, (doc, _) in enumerate(sorted_sparse, start=1):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
        
    # 3. Sort by RRF score descending
    sorted_contents = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top_n documents
    return [doc_map[content] for content, _ in sorted_contents[:top_n]]
