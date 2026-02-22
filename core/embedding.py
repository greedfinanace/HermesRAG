from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # SentenceTransformer handles CPU optimization internally
        logger.info("Embedding model loaded successfully.")

    def embed_chunks(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of text chunks in batches."""
        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = []
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
        if not embeddings:
            # Return empty array with correct shape if no chunks
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
            
        return np.vstack(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns 2D array for FAISS."""
        embedding = self.model.encode(query, show_progress_bar=False)
        # FAISS expects 2D array: (1, dim)
        return np.array([embedding], dtype=np.float32)
