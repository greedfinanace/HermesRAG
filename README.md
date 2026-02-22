# Offline Local RAG System

A production-ready, fully offline Retrieval-Augmented Generation (RAG) system running entirely on your local machine with zero cloud dependencies.

## Features
- **Offline LLM**: Integrates with [Ollama](https://ollama.com/) for local inference (e.g., Llama 3, Mistral).
- **Local Embeddings**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for fast, local vectorization.
- **Local Vector DB**: Uses `FAISS` with local disk persistence for similarity search.
- **Memory Efficient**: Batched chunk embedding, avoiding loading gigabytes of raw text into memory simultaneously.
- **Incremental Indexing**: Keeps track of file hashes. Adding new `.txt` files into the data directory only processes the new files, saving massive amounts of compute time.
- **Strict Grounding**: Prompt engineering strictly enforces the LLM to only use provided context.

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [Ollama's website](https://ollama.com/) and download a model:
   ```bash
   ollama run llama3
   ```
   *(Ensure Ollama is running in the background).*

## Setup

1. CD into the directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your `.txt` files. They default to being stored in `data/docs/`.
   ```bash
   # Or copy existing txt files manually
   ```

## Usage

### 1. Ingestion (Indexing)
Run the ingest command to process, chunk, embed, and store all documents in the target directory. 
```bash
python main.py ingest
```
*You can specify a different directory with `--dir /path/to/my/docs`.*

Incremental runs of this command will only process **new** or **modified** files.

### 2. Querying
Ask a question. The system will retrieve the top-K chunks and pass them to Ollama for a targeted answer.
```bash
python main.py query "What is the main topic of document X?"
```
*You can control how many chunks are retrieved with `--k 5`.*

## System Architecture

- `core/ingestion.py`: Handles character-level chunking with overlap and MD5 file hashing for state tracking.
- `core/embedding.py`: Uses `sentence-transformers` to turn raw text chunks into NumPy array vectors.
- `core/vector_store.py`: Wraps `FAISS` and manages the SQLite/JSON registry that maps vectors back to text chunks.
- `core/llm.py`: Connects to `Ollama` REST API, orchestrating the strict RAG prompt.
- `core/rag.py`: The pipeline gluing it all together.

## Optimization Advice for Low RAM Systems

If you are running this on a system with highly constrained RAM (e.g., <8GB) and parsing gigabytes of text:

1. **Reduce Batch Size**: In `core/embedding.py`, reduce `batch_size` in `embed_chunks()` from 32 down to 16 or 8. This controls how many strings are passed to the embedding model at once.
2. **JSON Lines Store**: Currently, `chunk_store.json` loads the entire chunk metadata into memory. For true gigabyte-scale datasets on 8GB RAM, modify `core/vector_store.py` to use SQLite (`sqlite3`) or a memory-mapped database instead of loading a list of dicts.
3. **FAISS Index Type**: `IndexFlatL2` stores raw vectors. At 10+ million chunks, switch to an IVFPQ index (`faiss.IndexIVFPQ`) which heavily compresses vectors using product quantization, effectively enabling 10x-100x the capacity in RAM.

## Performance Considerations for Large Document Sets

1. **Process Time**: CPU embedding of gigabytes of text will take hours. Leave `python main.py ingest` running overnight if ingesting huge datasets.
2. **Chunk Size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py` based on your document density. Smaller chunks (e.g., 500 chars) yield better precision for specific facts, while larger chunks (1500 chars) help with summarization queries. Ensure `TOP_K` is balanced against LLM context window limits.
