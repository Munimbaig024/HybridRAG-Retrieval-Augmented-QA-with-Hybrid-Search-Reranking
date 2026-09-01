import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Retrieval configurations
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "hybridrag_collection"

# Generation configurations
# Using Qwen on Groq
GROQ_MODEL = "qwen/qwen3.6-27b"

# Data processing
CHUNK_SIZE = 300 # in words
CHUNK_OVERLAP = 30 # in words
