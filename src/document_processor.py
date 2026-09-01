import wikipedia
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wikipedia often blocks requests without a custom user agent, returning HTML instead of JSON
wikipedia.set_user_agent("HybridRAG/1.0 (https://github.com/example/HybridRAG)")

def fetch_wikipedia_articles(topic: str, max_results: int = 10):
    """Fetches articles from wikipedia related to a topic."""
    logger.info(f"Fetching {max_results} articles for topic: {topic}")
    search_results = wikipedia.search(topic, results=max_results)
    
    articles = []
    for title in search_results:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            articles.append({
                "title": page.title,
                "text": page.content,
                "url": page.url
            })
            logger.info(f"Fetched article: {title}")
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Disambiguation error for {title}, skipping.")
        except wikipedia.exceptions.PageError as e:
            logger.warning(f"Page error for {title}, skipping.")
        except Exception as e:
            logger.error(f"Error fetching {title}: {e}")
            
    return articles

def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """Basic word-level chunking."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max(1, chunk_size - chunk_overlap)):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def process_articles_into_chunks(articles, chunk_size=300, chunk_overlap=30):
    """Takes a list of articles and returns a list of dictionaries with chunked text and metadata."""
    processed_chunks = []
    for idx, article in enumerate(articles):
        chunks = chunk_text(article["text"], chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "id": f"doc_{idx}_chunk_{chunk_idx}",
                "text": chunk,
                "metadata": {
                    "title": article["title"],
                    "url": article["url"]
                }
            })
    logger.info(f"Processed {len(articles)} articles into {len(processed_chunks)} chunks.")
    return processed_chunks
