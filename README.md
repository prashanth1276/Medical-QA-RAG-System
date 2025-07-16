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
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
ollama pull llama3
