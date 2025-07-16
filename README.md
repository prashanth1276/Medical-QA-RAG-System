# Medical QA RAG System

## Features
- PDF document ingestion with optimized chunking
- FAISS vector store with MMR reranking
- Llama3 via Ollama for answer generation
- Gradio web interface + CLI support
- Query caching system

## Installation
```bash
git clone https://github.com/yourusername/medical-qa-rag.git
cd medical-qa-rag
conda create -p venv python==3.12
conda activate venv/   # Windows
pip install -r requirements.txt
ollama pull llama3
ollama run llama3    # Run this in different Terminal window
