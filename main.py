import argparse
from pathlib import Path
import sys

from config import DOCS_DIR, TOP_K
from core.rag import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_parser.add_argument(
        "--dir", 
        type=str, 
        default=str(DOCS_DIR), 
        help=f"Directory containing .txt files (default: {DOCS_DIR})"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument(
        "question", 
        type=str, 
        help="The question to ask"
    )
    query_parser.add_argument(
        "--k", 
        type=int, 
        default=TOP_K, 
        help=f"Number of context chunks to retrieve (default: {TOP_K})"
    )

    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        pipeline = RAGPipeline()
        
        if args.command == "ingest":
            docs_path = Path(args.dir)
            pipeline.ingest(docs_path)
            
        elif args.command == "query":
            print("\n" + "="*50)
            print(f"Question: {args.question}")
            print("="*50)
            
            answer = pipeline.query(args.question, top_k=args.k)
            
            print("\nAnswer:")
            print("-" * 50)
            print(answer)
            print("-" * 50 + "\n")
            
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
