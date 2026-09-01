import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_reciprocal_rank(qrels: dict, run: dict, k: int = 10) -> float:
    """
    Computes MRR@k.
    qrels: dict mapping query_id to a list of relevant document IDs.
    run: dict mapping query_id to a list of retrieved document IDs (in rank order).
    """
    mrr_sum = 0.0
    queries_count = len(qrels)
    
    for qid, rel_docs in qrels.items():
        retrieved_docs = run.get(qid, [])[:k]
        for rank, doc_id in enumerate(retrieved_docs):
            if doc_id in rel_docs:
                mrr_sum += 1.0 / (rank + 1)
                break # Only the first relevant document matters for MRR
                
    return mrr_sum / queries_count if queries_count > 0 else 0.0


def precision_at_k(qrels: dict, run: dict, k: int = 10) -> float:
    """Computes Precision@k."""
    p_sum = 0.0
    queries_count = len(qrels)
    
    for qid, rel_docs in qrels.items():
        retrieved_docs = run.get(qid, [])[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in rel_docs)
        p_sum += relevant_retrieved / k
        
    return p_sum / queries_count if queries_count > 0 else 0.0

def recall_at_k(qrels: dict, run: dict, k: int = 10) -> float:
    """Computes Recall@k."""
    r_sum = 0.0
    queries_count = len(qrels)
    
    for qid, rel_docs in qrels.items():
        if not rel_docs:
            continue
        retrieved_docs = run.get(qid, [])[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in rel_docs)
        r_sum += relevant_retrieved / len(rel_docs)
        
    return r_sum / queries_count if queries_count > 0 else 0.0

def evaluate_run(qrels_file: str, run_file: str, k: int = 10):
    """Evaluates a retrieval run against qrels."""
    with open(qrels_file, 'r') as f:
        qrels = json.load(f)
        
    with open(run_file, 'r') as f:
        run = json.load(f)
        
    mrr = mean_reciprocal_rank(qrels, run, k)
    p_at_k = precision_at_k(qrels, run, k)
    r_at_k = recall_at_k(qrels, run, k)
    
    print(f"--- Evaluation Results (k={k}) ---")
    print(f"MRR@{k}:       {mrr:.4f}")
    print(f"Precision@{k}: {p_at_k:.4f}")
    print(f"Recall@{k}:    {r_at_k:.4f}")

if __name__ == "__main__":
    # Example usage: (You need to generate these files first)
    # evaluate_run("qrels.json", "run_results.json", k=5)
    print("Evaluation script loaded. To run, provide qrels and run JSON files.")
