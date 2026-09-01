from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name: str):
        """Initializes the cross-encoder model."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reranks a list of candidate documents using the cross-encoder.
        
        Args:
            query: The search query.
            candidates: List of document dictionaries (expected to have a "text" key).
            top_k: Number of top documents to return after reranking.
            
        Returns:
            A reranked and truncated list of document dictionaries.
        """
        if not candidates:
            return []
            
        # Prepare inputs for cross-encoder: list of (query, document_text) pairs
        query_doc_pairs = [[query, doc["text"]] for doc in candidates]
        
        # Get scores from the cross-encoder
        scores = self.model.predict(query_doc_pairs)
        
        # Sort candidates by the cross-encoder scores
        # argsort returns indices in ascending order, so we reverse them
        sorted_indices = np.argsort(scores)[::-1]
        
        reranked_results = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            doc = candidates[idx].copy()
            doc["cross_encoder_score"] = float(scores[idx])
            reranked_results.append(doc)
            
        return reranked_results
