import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"

# Create absolute paths for necessary directories
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Paths to artifacts
INDEX_PATH = INDEX_DIR / "vector_store.faiss"
STORE_PATH = INDEX_DIR / "chunk_store.json"
STATE_PATH = INDEX_DIR / "index_state.json"

# Semantic Chunking parameters
CHUNK_SIZE = 1500      # Soft limit characters per chunk
CHUNK_OVERLAP = 300    # overlap in characters
MIN_CHUNK_SIZE = 10    # minimum size to be considered a valid chunk

# Model parameters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434/api"

# Dynamically read OLLAMA_MODEL from environment variables, fallback to qwen2.5:3b
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()

# Supported File Extensions
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".html"}

# Retrieval parameters
TOP_K = 7              # Retrieve a bit more context for structural extraction
