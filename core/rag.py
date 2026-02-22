from pathlib import Path
from typing import List

from core.ingestion import load_documents, process_document
from core.embedding import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline components...")
        self.embedding_service = EmbeddingService()
        # Initialize VectorStore AFTER EmbeddingService so we can dynamically get dimension
        dimension = self.embedding_service.model.get_sentence_embedding_dimension()
        self.vector_store = VectorStore(dimension=dimension)
        self.llm_service = LLMService()
        logger.info("RAG Pipeline ready.")

    def ingest(self, docs_dir: Path):
        """Process new or updated documents in the directory."""
        logger.info(f"Scanning {docs_dir} for documents...")
        files_info = load_documents(docs_dir)
        
        if not files_info:
            logger.info("No documents found.")
            return

        all_new_chunks = []
        updated_state = {}
        
        for filename, info in files_info.items():
            if self.vector_store.is_file_processed(filename, info['hash']):
                logger.debug(f"Skipping {filename}: already processed.")
                continue
                
            logger.info(f"Processing new/updated file: {filename}")
            chunks = process_document(info['path'])
            if chunks:
                all_new_chunks.extend(chunks)
                updated_state[filename] = info['hash']
                
        if not all_new_chunks:
            logger.info("No new content to embed.")
            return
            
        logger.info(f"Generating embeddings for {len(all_new_chunks)} new chunks...")
        texts = [c['text'] for c in all_new_chunks]
        embeddings = self.embedding_service.embed_chunks(texts)
        
        self.vector_store.add_embeddings(embeddings, all_new_chunks, updated_state)
        logger.info("Ingestion complete.")

    def query(self, user_query: str, top_k: int = 5) -> str:
        """Answer a question based on ingested documents."""
        logger.info(f"Processing query: '{user_query}'")
        
        # 1. Embed query
        query_embedding = self.embedding_service.embed_query(user_query)
        
        # 2. Search vector store
        logger.info(f"Retrieving top {top_k} chunks...")
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not retrieved_chunks:
            return "No relevant documents have been ingested yet. Please add documents and run ingestion."
            
        # 3. Generate response with LLM
        response = self.llm_service.generate_response(user_query, retrieved_chunks)
        
        if response is None:
            return "Failed to generate a response due to an LLM error. Check logs."
            
        return response
