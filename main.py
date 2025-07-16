from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.interface import create_gradio_interface
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def build_system():
    """End-to-end pipeline builder"""
    processor = DocumentProcessor()
    rag = RAGPipeline()
    
    # Ingest documents
    chunks = processor.load_and_chunk("data/9241544228_eng.pdf")
    rag.ingest_documents(chunks)
    
    return rag

if __name__ == "__main__":
    system = build_system()
    iface = create_gradio_interface(system)
    iface.launch()