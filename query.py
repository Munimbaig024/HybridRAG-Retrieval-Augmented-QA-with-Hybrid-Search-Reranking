import logging
from src.pipeline import HybridRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Initializing Pipeline ---")
    pipeline = HybridRAGPipeline()
    
    logger.info("--- Loading State (No internet fetching needed!) ---")
    try:
        pipeline.load_state()
    except FileNotFoundError:
        logger.error("Could not find saved state. Please run `python ingest.py` first to build the index.")
        return
        
    print("\n" + "="*50)
    print("Welcome to HybridRAG! (Type 'quit' or 'exit' to stop)")
    print("="*50)
    
    while True:
        try:
            q = input("\nEnter your query: ")
        except EOFError:
            break
            
        if q.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not q.strip():
            continue
            
        print(f"\nSearching for: {q}...")
        result = pipeline.run(query=q)
        
        print("\n--- Answer ---")
        print(result["answer"])
        
        print("\n--- Sources Context ---")
        for i, ctx in enumerate(result["context"]):
            title = ctx["metadata"]["title"]
            score = ctx.get("cross_encoder_score", "N/A")
            print(f"[{i+1}] {title} (Rerank Score: {score:.4f})")

if __name__ == "__main__":
    main()
