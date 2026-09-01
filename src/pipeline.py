import logging
from src.retrievers import DenseRetriever, SparseRetriever
from src.fusion import reciprocal_rank_fusion
from src.reranker import Reranker
from src.generator import AnswerGenerator
from src import config

logger = logging.getLogger(__name__)

class HybridRAGPipeline:
    def __init__(self):
        """Initializes all components of the HybridRAG pipeline."""
        logger.info("Initializing Dense Retriever...")
        self.dense_retriever = DenseRetriever(
            db_dir=config.CHROMA_DB_DIR,
            collection_name=config.COLLECTION_NAME,
            model_name=config.DENSE_MODEL_NAME
        )
        
        logger.info("Initializing Sparse Retriever...")
        self.sparse_retriever = SparseRetriever()
        
        logger.info("Initializing Cross-Encoder Reranker...")
        self.reranker = Reranker(model_name=config.CROSS_ENCODER_MODEL_NAME)
        
        logger.info("Initializing Answer Generator (Groq)...")
        self.generator = AnswerGenerator(model_name=config.GROQ_MODEL)
        
        self.is_fitted = False

    def ingest_documents(self, documents: list[dict]):
        """Ingests chunks into both retrievers."""
        logger.info(f"Ingesting {len(documents)} documents into vector store...")
        self.dense_retriever.add_documents(documents)
        
        logger.info("Fitting BM25 model...")
        self.sparse_retriever.fit(documents)
        
        self.is_fitted = True
        logger.info("Ingestion complete.")

    def save_state(self, filepath: str = "sparse_index.pkl"):
        """Saves the sparse retriever state. (Dense retriever is already persistent)"""
        logger.info(f"Saving sparse index to {filepath}...")
        self.sparse_retriever.save(filepath)

    def load_state(self, filepath: str = "sparse_index.pkl"):
        """Loads the sparse retriever state."""
        logger.info(f"Loading sparse index from {filepath}...")
        self.sparse_retriever.load(filepath)
        self.is_fitted = True

    def run(self, query: str, dense_k: int = 10, sparse_k: int = 10, final_k: int = 5) -> dict:
        """
        Runs the full HybridRAG pipeline for a given query.
        
        Args:
            query: The user's query.
            dense_k: Top K results to fetch from dense retriever.
            sparse_k: Top K results to fetch from sparse retriever.
            final_k: Top K results to send to the LLM after reranking.
            
        Returns:
            A dictionary containing the generated answer and the intermediate contexts.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted with documents before running queries.")
            
        logger.info(f"Processing query: '{query}'")
        
        # 1. Retrieval
        logger.info("Running Dense Retrieval...")
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)
        
        logger.info("Running Sparse Retrieval (BM25)...")
        sparse_results = self.sparse_retriever.retrieve(query, top_k=sparse_k)
        
        # 2. Fusion
        logger.info("Fusing results using RRF...")
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
        
        # 3. Reranking
        logger.info("Reranking top candidates with Cross-Encoder...")
        # Get top 50 (or less if fewer fused results) for reranking to save time
        candidates_for_reranking = fused_results[:50] 
        reranked_results = self.reranker.rerank(query, candidates_for_reranking, top_k=final_k)
        
        # 4. Generation
        logger.info("Generating answer with Groq LLM...")
        answer = self.generator.generate_answer(query, reranked_results)
        
        return {
            "query": query,
            "answer": answer,
            "context": reranked_results,
            "debug": {
                "dense_results": dense_results,
                "sparse_results": sparse_results,
                "fused_results": fused_results
            }
        }
