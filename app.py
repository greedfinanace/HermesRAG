from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from pathlib import Path

from core.rag import RAGPipeline
from config import DOCS_DIR, TOP_K, SUPPORTED_EXTENSIONS
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Offline RAG")

os.makedirs(DOCS_DIR, exist_ok=True)
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Initialize RAG Pipeline on startup
try:
    pipeline = RAGPipeline()
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    pipeline = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if pipeline is None:
        return {"error": "Pipeline failed to initialize during startup."}
        
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"}
        
    file_location = os.path.join(DOCS_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger ingestion synchronously to prevent race conditions
        pipeline.ingest(Path(DOCS_DIR))
        return {"message": f"File '{file.filename}' successfully uploaded and indexed."}
    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return {"error": f"Server error handling upload: {str(e)}"}

@app.post("/query/")
async def query(question: str = Form(...)):
    if pipeline is None:
        return {"error": "Pipeline failed to initialize. Check server logs."}
        
    if not question.strip():
        return {"error": "Question cannot be empty"}
        
    try:
        # Pipeline now returns structured dict or a string depending on state
        answer = pipeline.query(question, top_k=TOP_K)
        
        # If RAG returns a plain string (e.g. "No documents ingested")
        if isinstance(answer, str):
            if "No relevant documents" in answer:
                return {"error": answer}
            # Fallback for unexpected string returns
            return {
                "answer": {
                    "main_claim": answer,
                    "supporting_evidence": "N/A",
                    "methodology": "N/A",
                    "cited_references": "N/A"
                }
            }
            
        # If it returns an error dict from LLM
        if "error" in answer:
            return {"error": answer["error"]}
            
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return {"error": f"Internal server error while processing query: {str(e)}"}
