import argparse
from src.pipeline import ingest_pdf_to_faiss

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Pipeline CLI")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--use-ocr", action="store_true", help="Enable OCR processing")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM metadata extraction")
    parser.add_argument("--output", type=str, default="vectorstore/faiss_index", help="Output directory for FAISS index")
    
    args = parser.parse_args()
    
    print(f"[START] Starting ingestion for {args.pdf}")
    stats = ingest_pdf_to_faiss(
        pdf_path=args.pdf,
        index_dir=args.output,
        use_ocr_if_needed=args.use_ocr,
        use_llm_metadata=args.use_llm
    )
    print("[DONE] Pipeline complete!")
    print(stats)

if __name__ == "__main__":
    main()
