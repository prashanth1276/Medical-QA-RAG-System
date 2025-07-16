from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor

def test_medical_queries(rag_pipeline):
    test_cases = [
        {
            "query": "What is the ICD-10 code for recurrent depressive disorder in remission?",
            "expected": "F33.4",
            "type": "code"
        },
        {
            "query": "Explain the diagnostic criteria for bipolar disorder",
            "expected": ["episode", "mania", "depression"],
            "type": "general"
        },
        {
            "query": "What medications treat anxiety disorders?",
            "expected": ["antidepressants", "benzodiazepines"],
            "type": "general"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['query']}")
        answer = rag_pipeline.answer_query(case['query'])
        
        if case["type"] == "code":
            # Verify code extraction
            code_match = re.search(r'[A-Z]\d{2}\.\d', answer)
            result = "PASS" if code_match else "FAIL"
            print(f"Answer: {answer} | {result}")
        else:
            # Verify content presence
            result = "PASS" if all(term.lower() in answer.lower() for term in case["expected"]) else "FAIL"
            print(f"Answer: {answer[:200]}... | {result}")

if __name__ == "__main__":
    print("Building verification pipeline...")
    rag_pipeline = RAGPipeline()
    rag_pipeline.ingest_documents(DocumentProcessor().load_and_chunk("data/9241544228_eng.pdf"))
    
    print("\nRunning verification tests...")
    test_medical_queries(rag_pipeline)