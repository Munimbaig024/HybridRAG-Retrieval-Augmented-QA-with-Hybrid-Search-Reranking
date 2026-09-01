import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np

class DenseRetriever:
    def __init__(self, db_dir: str, collection_name: str, model_name: str):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: list[dict]):
        """Adds documents to the chroma collection."""
        if not documents:
            return
            
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query: str, top_k: int = 10):
        """Retrieves top_k chunks based on semantic similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                })
        return retrieved


class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def fit(self, documents: list[dict]):
        """Fits the BM25 model on the given documents."""
        self.documents = documents
        
        # Simple tokenization by splitting on whitespace. 
        # For production, consider using nltk or spacy.
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def save(self, filepath: str):
        """Saves the BM25 model and documents to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'documents': self.documents}, f)
            
    def load(self, filepath: str):
        """Loads the BM25 model and documents from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.documents = data['documents']

    def retrieve(self, query: str, top_k: int = 10):
        """Retrieves top_k chunks based on lexical matching (BM25)."""
        if self.bm25 is None:
            raise ValueError("SparseRetriever has not been fitted with documents yet.")
            
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = []
        for idx in top_n_indices:
            # Only include if score > 0 (to avoid retrieving completely irrelevant docs)
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                retrieved.append(doc)
                
        return retrieved
