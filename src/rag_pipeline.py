import time
from src.vector_store import VectorStore
import ollama
import re
from typing import List
import logging
from functools import lru_cache

class RAGPipeline:
    """Universal medical QA system using RAG with Llama3."""
    
    def __init__(self, model_name: str = "llama3"):
        self.vector_store = VectorStore()
        self.model_name = model_name
        # Pre-load model during initialization
        self.client = ollama.Client()
        self.logger = logging.getLogger(__name__)
        
        # Add model readiness check here 
        self._wait_for_model_ready()  # Ensures model is loaded and ready

    def _wait_for_model_ready(self):
        """Ensures Ollama model is fully loaded before proceeding"""
        while True:
            try:
                self.client.pull(self.model_name)  # Ensure model exists
                if ollama.list():  # Verify model is loaded
                    self.logger.info(f"Model {self.model_name} ready")
                    break
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Model loading attempt failed: {str(e)}")
                time.sleep(1)

    def ingest_documents(self, chunks: List[str]):
        """Store document chunks with enhanced medical text processing"""
        medical_chunks = [self._preprocess_medical_text(c) for c in chunks]
        self.vector_store.add_documents(medical_chunks)

    @lru_cache(maxsize=100)
    def answer_query(self, query: str) -> str:
        """Handle any medical query with dynamic context retrieval"""
        try:
            # Adaptive context window
            k = 5 if len(query.split()) > 10 else 3  # More context for complex queries
            context_chunks = self.vector_store.search(query, k=k)
            context = "\n\n---\n".join([chunk for chunk, _ in context_chunks])
            
            prompt = f"""You are a medical expert. Answer based ONLY on:
                        
Context:
{context}

Question:
{query}

Provide a concise, accurate response. If unsure, say "Not covered in guidelines"."""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.2,
                         'timeout': 10  # 10-second timeout
                        }
            )
            
            return self._postprocess(response['response'])
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return "System error. Please try again."

    def _preprocess_medical_text(self, text: str) -> str:
        """Clean medical text for better retrieval"""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)  # Fix line breaks
        return text.strip()

    def _postprocess(self, response: str) -> str:
        """Clean and validate responses"""
        response = response.split("Question:")[0].strip()
        return re.sub(r'\[.*?\]', '', response)  # Remove citations