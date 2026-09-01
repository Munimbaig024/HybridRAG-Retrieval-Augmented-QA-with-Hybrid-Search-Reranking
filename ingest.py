import logging
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.document_processor import fetch_wikipedia_articles, process_articles_into_chunks
from src.pipeline import HybridRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    topic = "Machine learning"
    logger.info(f"--- Fetching data for topic '{topic}' ---")
    articles = fetch_wikipedia_articles(topic, max_results=5)
    
    logger.info("--- Chunking data ---")
    chunks = process_articles_into_chunks(articles, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    logger.info("--- Initializing Pipeline ---")
    pipeline = HybridRAGPipeline()
    
    logger.info("--- Ingesting data ---")
    pipeline.ingest_documents(chunks)
    
    logger.info("--- Saving State ---")
    pipeline.save_state()
    
    logger.info("Ingestion complete! Data is saved locally. You can now run `python query.py` repeatedly without re-fetching data.")

if __name__ == "__main__":
    main()
