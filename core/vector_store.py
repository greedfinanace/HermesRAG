import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import INDEX_PATH, STORE_PATH, STATE_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, dimension: int = 384): # Default for all-MiniLM-L6-v2
        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Please install it to use VectorStore.")
            
        self.dimension = dimension
        self.index = None
        self.chunk_store = [] # List mapping FAISS internal ID to chunk data
        self.doc_state = {}   # Mapping of filename -> hash to track processed files
        
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index and stores from disk, or create new ones."""
        index_loaded = False
        if os.path.exists(INDEX_PATH):
            try:
                self.index = faiss.read_index(str(INDEX_PATH))
                logger.info(f"Loaded FAISS index from {INDEX_PATH} with {self.index.ntotal} vectors.")
                index_loaded = True
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                
        if not index_loaded:
            logger.info("Creating new FAISS IndexFlatL2.")
            self.index = faiss.IndexFlatL2(self.dimension)
            
        if os.path.exists(STORE_PATH):
            try:
                with open(STORE_PATH, 'r', encoding='utf-8') as f:
                    self.chunk_store = json.load(f)
                logger.info(f"Loaded chunk store with {len(self.chunk_store)} items.")
            except Exception as e:
                logger.error(f"Failed to load chunk store: {e}")
                self.chunk_store = []
                
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, 'r', encoding='utf-8') as f:
                    self.doc_state = json.load(f)
                logger.info(f"Loaded document state matching {len(self.doc_state)} files.")
            except Exception as e:
                logger.error(f"Failed to load doc state: {e}")
                self.doc_state = {}
                
        # Basic integrity check
        if self.index.ntotal != len(self.chunk_store):
            logger.warning(f"Index size ({self.index.ntotal}) and chunk store size ({len(self.chunk_store)}) mismatch!")

    def save(self):
        """Persist index, store, and state to disk."""
        logger.info("Saving vector store to disk...")
        faiss.write_index(self.index, str(INDEX_PATH))
        
        with open(STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_store, f, ensure_ascii=False)
            
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.doc_state, f, ensure_ascii=False)
            
        logger.info("Save complete.")

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]], updated_state: Dict[str, str]):
        """Add new embeddings to the index and update stores."""
        if len(embeddings) == 0:
            logger.info("No embeddings to add.")
            return
            
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
            
        self.index.add(embeddings)
        self.chunk_store.extend(chunks)
        self.doc_state.update(updated_state)
        
        logger.info(f"Added {len(chunks)} chunks to the store. Total chunks: {len(self.chunk_store)}.")
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index for the most similar chunks."""
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index.")
            return []
            
        # Ensure k is not greater than the total number of vectors
        k = min(top_k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        # Return chunks and their distances
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunk_store):
                result = dict(self.chunk_store[idx])
                result['distance'] = float(distances[0][i])
                results.append(result)
                
        return results

    def is_file_processed(self, filename: str, file_hash: str) -> bool:
        """Check if a file with the given hash has already been processed."""
        return self.doc_state.get(filename) == file_hash
