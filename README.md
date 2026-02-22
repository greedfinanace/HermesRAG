# HermesRAG

A fully offline, local Retrieval-Augmented Generation (RAG) system dedicated to analyzing research papers and raw datasets. It uses local LLMs (via Ollama) and FAISS vector databases to ensure complete data privacy.

## Features

- Fully Offline: No cloud APIs or data sharing.
- Multi-Format Support: Reads .pdf, .docx, .md, .txt, .csv, .json, and .html files out of the box.
- Semantic Chunking: Intelligently slices documents by paragraph boundaries to preserve context.
- Structured LLM Extraction: Forces the AI to output parsed insights (Main Claim, Evidence, Methodology, References) instead of rambling text.
- Simple Drag-and-Drop Web UI.
- Automated Windows Control Panel Launcher.

## Prerequisites

1. Install Python 3.10+
2. Install Ollama (Download from ollama.com)
3. Download at least one local AI model through your terminal. For example:
   `ollama run qwen2.5:3b`

## Setup Instructions

1. Clone or download this repository.
2. Open a terminal in the repository folder.
3. Install the required Python dependencies:
   `pip install -r requirements.txt`

## How to Run

1. Navigate to the project folder.
2. Double-click the `start_rag.bat` file.
3. The integrated terminal will prompt you to select the AI model you wish to use by typing a number.
4. The system will automatically ensure the Ollama engine is running, lease an open network port, and launch the RAG web application in your default browser.
5. Drag and drop your research documents onto the website interface to index them.
6. Type your research questions into the query bar to extract structured insights.
