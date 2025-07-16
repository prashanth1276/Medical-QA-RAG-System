import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
import logging

class VectorStore:
    """High-recall medical document retrieval system"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.index = None
        self.documents = []
        self.logger = logging.getLogger(__name__)

    def add_documents(self, chunks: List[str]):
        """Batch ingest with medical text optimization"""
        embeddings = self.model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Better for medical QA
        )
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        
        self.index.add(embeddings)
        self.documents.extend(chunks)
        self.logger.info(f"Added {len(chunks)} chunks")

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Semantic search with medical term boosting"""
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_tensor=True  # Faster than numpy for single queries
        )
        
        # Boost ICD-10 code relevance
        if re.search(r'[A-Z]\d{2}\.\d', query):
            k = min(k * 2, 10)  # Wider search for code queries
        
        scores, indices = self.index.search(query_embedding.numpy(), k)
        return [
            (self.documents[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if i != -1
        ]