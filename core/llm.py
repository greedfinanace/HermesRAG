import json
import requests
import re
from typing import List, Dict, Any, Optional

from config import OLLAMA_BASE_URL, LLM_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Strict prompt template forcing structured JSON-like grounding
RAG_PROMPT_TEMPLATE = """You are a precise, helpful AI research assistant. 
You will be provided with context from documents and a question.
Your task is to analyze the context and answer the question STRICTLY using ONLY the provided context. 

You MUST output your response exactly in the following structured XML format. Do not add conversational filler.

<response>
  <main_claim>Provide the primary answer or thesis here.</main_claim>
  <supporting_evidence>List the specific facts or data from the text supporting the claim.</supporting_evidence>
  <methodology>Describe how the information was derived, if mentioned in the text. Otherwise write "Not mentioned."</methodology>
  <cited_references>List the exact document sources used to form this answer.</cited_references>
</response>

If the answer is not contained within the context, output:
<response>
  <main_claim>I cannot answer this question based on the provided context.</main_claim>
  <supporting_evidence>N/A</supporting_evidence>
  <methodology>N/A</methodology>
  <cited_references>N/A</cited_references>
</response>

Context:
{context}

Question:
{question}

Answer:"""

class LLMService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = LLM_MODEL):
        self.base_url = base_url
        self.model = model
        
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format the retrieved chunks into a single string for the prompt."""
        if not retrieved_chunks:
            return "No relevant context found."
            
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk['metadata'].get('source', 'Unknown source')
            text = chunk.get('text', '')
            context_parts.append(f"--- Document [{i+1}] Source: {source} ---\n{text}\n")
            
        return "\n".join(context_parts)

    def _parse_xml_response(self, text: str) -> Dict[str, str]:
        """Robutsly parse the XML-like structured response from the LLM."""
        result = {
            "main_claim": "Failed to parse main claim.",
            "supporting_evidence": "Failed to parse evidence.",
            "methodology": "Failed to parse methodology.",
            "cited_references": "Failed to parse references."
        }
        
        tags = ["main_claim", "supporting_evidence", "methodology", "cited_references"]
        for tag in tags:
            # Use regex to find content between tags, handling potential newlines
            match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            if match:
                result[tag] = match.group(1).strip()
                
        # If regex utterly fails, dump the raw text into main_claim so the user sees *something*
        if result["main_claim"] == "Failed to parse main claim." and "<response>" not in text.lower():
            result["main_claim"] = text.strip()
            
        return result

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Generate a structured answer using Ollama and the retrieved context."""
        context = self._format_context(retrieved_chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        
        logger.info(f"Sending structured prompt to Ollama model {self.model}...")
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                    # Note: Not using format="json" because XML parsing is often more reliable
                    # for smaller 3b-4b parameter models that might forget to escape JSON quotes.
                },
                timeout=120 
            )
            response.raise_for_status()
            
            result_json = response.json()
            raw_text = result_json.get("response", "")
            
            return self._parse_xml_response(raw_text)
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {self.base_url}. Is it running?")
            return {"error": "Ollama connection failed. Ensure Ollama is running (`ollama serve`)."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return {"error": f"LLM Error: {str(e)}"}
