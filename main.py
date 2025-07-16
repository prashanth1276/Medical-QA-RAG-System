from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.interface import create_gradio_interface
import logging

# Basic config 
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Silence specific loggers
noisy_loggers = [
    "httpx", "urllib3", 
    "transformers", "sentence_transformers",
    "gradio.http_request"
]
for logger in noisy_loggers:
    logging.getLogger(logger).setLevel(logging.WARNING)


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