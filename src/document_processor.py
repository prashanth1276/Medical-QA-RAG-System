from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from typing import List
import logging

class DocumentProcessor:
    """Medical PDF processor with context-aware chunking"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n• ", "\n• ", "\n\n", "\n", "(?<=\. )", " "],
            keep_separator=True
        )
        self.logger = logging.getLogger(__name__)

    def load_and_chunk(self, file_path: str) -> List[str]:
        """PDF loader with medical document intelligence"""
        try:
            reader = PdfReader(file_path)
            text = "\n".join([
                self._clean_medical_page(p.extract_text() or "")
                for p in reader.pages
            ])
            
            chunks = self.splitter.split_text(text)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            raise

    def _clean_medical_page(self, text: str) -> str:
        """Medical text normalization"""
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Line breaks
        text = re.sub(r'\s+', ' ', text)  # Whitespace
        return text.strip()