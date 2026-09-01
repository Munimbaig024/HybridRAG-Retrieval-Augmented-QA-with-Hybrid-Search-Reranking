def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        dense_results: Ranked list of dictionaries from the dense retriever.
        sparse_results: Ranked list of dictionaries from the sparse retriever.
        k: The RRF constant (commonly 60).
        
    Returns:
        A combined and re-ranked list of unique documents.
    """
    # Dictionary to hold the accumulated RRF score for each document ID
    rrf_scores = {}
    
    # Dictionary to keep the actual document data
    doc_lookup = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        doc_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        doc_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort documents by their accumulated RRF score in descending order
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Reconstruct the ranked list of documents
    fused_results = []
    for doc_id, score in sorted_items:
        doc = doc_lookup[doc_id].copy()
        doc["rrf_score"] = score
        fused_results.append(doc)
        
    return fused_results
