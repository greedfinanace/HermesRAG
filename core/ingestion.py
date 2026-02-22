import os
import hashlib
import re
import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    markdown = None

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, SUPPORTED_EXTENSIONS
from utils.logger import get_logger

logger = get_logger(__name__)

def get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file to detect changes."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {filepath}: {e}")
        return ""

def text_to_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs to maintain semantic boundaries."""
    # Split by double newline or multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean up and filter empty
    return [p.strip() for p in paragraphs if p.strip()]

def chunk_text_semantically(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Split text into chunks, respecting paragraph boundaries where possible.
    Falls back to character-window chunking if a paragraph is too huge.
    """
    paragraphs = text_to_paragraphs(text)
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        p_len = len(p)
        
        # If a single paragraph is larger than our target CHUNK_SIZE, we must split it by sentences or characters
        if p_len > CHUNK_SIZE:
            # If we have accumulated text, save it first
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= MIN_CHUNK_SIZE:
                    chunks.append({"text": chunk_text, "metadata": {"source": source}})
                current_chunk = []
                current_length = 0
            
            # Split giant paragraph by character window
            start = 0
            while start < p_len:
                end = min(start + CHUNK_SIZE, p_len)
                sub_chunk = p[start:end]
                if len(sub_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append({"text": sub_chunk, "metadata": {"source": source}})
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
            continue
            
        # If adding this paragraph exceeds the chunk size, save current chunk and start new one with overlap
        if current_length + p_len > CHUNK_SIZE and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({"text": chunk_text, "metadata": {"source": source}})
            
            # Keep the last paragraph as overlap if it's not too big
            overlap_text = current_chunk[-1] if current_chunk else ""
            current_chunk = [overlap_text, p] if overlap_text else [p]
            current_length = len(overlap_text) + len(p)
        else:
            current_chunk.append(p)
            current_length += p_len + 2 # +2 for the \n\n
            
    # Add any remaining text
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({"text": chunk_text, "metadata": {"source": source}})
            
    return chunks

def extract_pdf_text(filepath: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    if fitz is None:
        logger.error("PyMuPDF (fitz) is not installed. Cannot process PDF.")
        return ""
    
    try:
        doc = fitz.open(filepath)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to read PDF {filepath}: {e}")
        return ""

def extract_docx_text(filepath: Path) -> str:
    """Extract text from a Word document."""
    if docx is None:
        logger.error("python-docx is not installed. Cannot process DOCX.")
        return ""
        
    try:
        doc = docx.Document(filepath)
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        logger.error(f"Failed to read DOCX {filepath}: {e}")
        return ""

def extract_md_text(filepath: Path) -> str:
    """Extract plain text from Markdown."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if markdown is not None:
            # Convert to HTML, then rip text out to avoid weird markdown artifacts
            html = markdown.markdown(text)
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
            
        # Fallback to plain text if bs4/markdown not available
        return text
    except Exception as e:
        logger.error(f"Failed to read MD {filepath}: {e}")
        return ""

def extract_txt_text(filepath: Path) -> str:
    """Extract text from a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read TXT {filepath}: {e}")
        return ""

def extract_csv_text(filepath: Path) -> str:
    """Extract text from a CSV file."""
    try:
        text_parts = []
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Join columns with a space, rows with a double newline to simulate paragraphs
                cleaned_row = [str(cell).strip() for cell in row if str(cell).strip()]
                if cleaned_row:
                    text_parts.append(" | ".join(cleaned_row))
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to read CSV {filepath}: {e}")
        return ""

def extract_json_text(filepath: Path) -> str:
    """Extract text from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert dictionary to a formatted string representation
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Failed to read JSON {filepath}: {e}")
        return ""

def extract_html_text(filepath: Path) -> str:
    """Extract text from an HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
            
        if markdown is not None:
             soup = BeautifulSoup(html, "html.parser")
             # Rip out script and style elements
             for script in soup(["script", "style"]):
                 script.extract()
             return soup.get_text(separator=' ', strip=True)
             
        return html
    except Exception as e:
        logger.error(f"Failed to read HTML {filepath}: {e}")
        return ""

def process_document(filepath: Path) -> List[Dict[str, Any]]:
    """Read a document based on its extension and return its chunks."""
    ext = filepath.suffix.lower()
    
    if ext == '.pdf':
        text = extract_pdf_text(filepath)
    elif ext == '.docx':
        text = extract_docx_text(filepath)
    elif ext == '.md':
        text = extract_md_text(filepath)
    elif ext == '.txt':
        text = extract_txt_text(filepath)
    elif ext == '.csv':
        text = extract_csv_text(filepath)
    elif ext == '.json':
        text = extract_json_text(filepath)
    elif ext == '.html':
        text = extract_html_text(filepath)
    else:
        logger.warning(f"Unsupported file extension for {filepath.name}")
        return []
        
    if not text.strip():
        logger.warning(f"Extracted no text from {filepath.name}")
        return []
        
    return chunk_text_semantically(text, source=filepath.name)

def load_documents(docs_dir: Path) -> Dict[str, Any]:
    """
    Scan directory for supported files.
    Returns mapping of filename to its hash and full path.
    """
    files_info = {}
    if not docs_dir.exists():
        logger.warning(f"Directory {docs_dir} does not exist.")
        return files_info
        
    for file_path in docs_dir.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            file_hash = get_file_hash(str(file_path))
            if file_hash:
                files_info[file_path.name] = {
                    "path": file_path,
                    "hash": file_hash
                }
    return files_info
