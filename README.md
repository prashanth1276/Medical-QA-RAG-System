# Medical QA RAG System

## Overview
A RAG-based question answering system for medical guidelines documentation, implemented within a 4-hour timeframe as per the WundrSight SWE Intern technical assignment. The system processes the ICD-10 Classification of Mental and Behavioural Disorders (9241544228_eng.pdf, 377 pages) to provide coded classifications for diagnoses.

## Tools and Models Used
- **Embeddings**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Vector DB**: FAISS (CPU version, IndexFlatIP)
- **LLM**: LLaMA3 via Ollama
- **Text Processing**: PyPDF2, LangChain RecursiveCharacterTextSplitter
- **Interface**: Gradio

## System Components
| File | Description |
|------|-------------|
| `document_processor.py` | PDF text extraction and medical-optimized chunking (500 tokens, 100 overlap) |
| `vector_store.py` | FAISS index management with ICD-10 code boosting |
| `rag_pipeline.py` | Core RAG logic with query caching via lru_cache |
| `interface.py` | Gradio web interface |
| `main.py` | Entry point with system initialization and document ingestion |

## Key Design Decisions
1. **Chunking Strategy**:
   - 500-token chunks with 100-token overlap to retain context in a 377-page medical text with dense descriptions and ICD-10 codes.
   - Medical-aware separators (e.g., bullet points, section breaks, periods) to preserve clinical structure.
   - Line break normalization to handle PDF formatting artifacts.

2. **Retrieval Optimization**:
   - Boosted search width for ICD-10 code queries
   - Cosine similarity with normalized embeddings

3. **LLM Interaction**:
   - Dynamic context window (k=3 for short queries, k=5 for complex ones) based on query length.
   - Temperature=0.2 for factual, concise responses.
   - Strict context-bound answers with a prompt limiting responses to retrieved chunks.
     
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
   - Processes only the provided 9241544228_eng.pdf (ICD-10 guidelines, 377 pages); no dynamic document uploads or external data integration.

2. **Code Coverage**:
   - No Maximal Marginal Relevance (MMR) reranking implemented due to 4-hour time constraint.
   - Error handling is limited to generic messages (e.g., "System error. Please try again") with basic logging; lacks specific feedback for edge cases like missing chunks or LLM failures.

3. **Performance**:
   - Initial query latency ~15-30 seconds due to embedding generation and LLM inference for a 377-page document.
   - Requires local Ollama setup with LLaMA3 pre-loaded for operation.

     
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
# Get llama3 (optional)
ollama pull llama3

# Launch system
python main.py
