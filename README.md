# Medical QA RAG System

## Overview
A RAG-based question answering system for medical guidelines documentation, implemented within a 4-hour timeframe as per technical assignment requirements.

## Tools and Models Used
- **Embeddings**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Vector DB**: FAISS (CPU version)
- **LLM**: Llama3 via Ollama
- **Text Processing**: PyPDF2, LangChain text splitter
- **Interface**: Gradio

## System Components
| File | Description |
|------|-------------|
| `document_processor.py` | PDF text extraction and medical-optimized chunking |
| `vector_store.py` | FAISS index management with medical term boosting |
| `rag_pipeline.py` | Core RAG logic with query caching |
| `interface.py` | Gradio web interface |
| `main.py` | Entry point with system initialization |

## Key Design Decisions
1. **Chunking Strategy**:
   - 500-character chunks with 100-character overlap
   - Medical-aware separators (bullet points, section breaks)
   - Line break normalization for clinical text

2. **Retrieval Optimization**:
   - Boosted search width for ICD-10 code queries
   - Cosine similarity with normalized embeddings

3. **LLM Interaction**:
   - Dynamic context window based on query complexity
   - Temperature=0.2 for factual responses
   - Strict context-bound answers
     
4. **Caching**:
   - Implemented LRU cache for frequent queries
   - Assumed repeated queries would be common in clinical use

## Sample Output
**Test Query**:  
```text
Give me the correct coded classification for: "Recurrent depressive disorder, currently in remission"
```

**Output**: 
```text
The correct coded classification is:

F33.4 Recurrent depressive disorder, currently in remission
```

## Limitations
1. **Document Scope**:
   - Only processes the provided clinical guidelines PDF
   - No web/document crawling capability

2. **Code Coverage**:
   - ICD-10 validation limited to mood disorders
   - No cross-coding system support (e.g., SNOMED)

3. **Performance**:
   - Initial query latency ~15-30 seconds
   - Requires local Ollama setup

## AI Tools Usage
- **ChatGPT**:
  - Used for debugging FAISS index configuration
  - Suggested optimization for medical text preprocessing regex patterns
- **Grok AI**:
  - Assisted in designing the prompt engineering strategy for Llama3
  - For Interface
  - Recommended the chunk size (500 chars) based on medical text characteristics
  - Provided insights on error handling patterns for Medical QA RAG system


## How to Run
```bash
# Clone the Repo
git clone https://github.com/prashanth1276/Medical-QA-RAG-System.git
cd Medical-QA-RAG-System

# Set up environment
conda create -p venv python==3.12   #Windows
conda activate venv/    # Windows

# Install all the Dependencies
pip install -r requirements.txt

# First Install Ollama in your system
# Get llama3
ollama pull llama3
# Run llama3
ollama run llama3

# Launch system
python main.py
