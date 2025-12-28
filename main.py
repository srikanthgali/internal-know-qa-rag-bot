"""
Main entry point for RAG chatbot.

This script provides a unified interface to run different components.

Usage:
    python main.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_index(data_dir: str):
    """Build FAISS index from documents."""
    logger.info("Building FAISS index...")
    from src.embeddings.build_index import IndexBuilder

    builder = IndexBuilder()
    builder.build_index_from_directory(data_dir)
    logger.info("✓ Index built successfully")


def run_api():
    """Run FastAPI server."""
    logger.info("Starting API server...")
    subprocess.run([sys.executable, "run_api.py"])


def run_ui():
    """Run Streamlit UI."""
    logger.info("Starting Streamlit UI...")
    subprocess.run([sys.executable, "run_streamlit.py"])


def ingest_documents(url: str, max_pages: int, output_dir: str):
    """Ingest documents from URL."""
    logger.info(f"Ingesting documents from {url}")
    subprocess.run(
        [
            sys.executable,
            "scripts/ingest_documents.py",
            "--url",
            url,
            "--max-pages",
            str(max_pages),
            "--output",
            output_dir,
        ]
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Internal Knowledge Base RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from documents
  python main.py build-index --data-dir data/raw

  # Run API server
  python main.py api

  # Run Streamlit UI
  python main.py ui

  # Ingest documents
  python main.py ingest --url https://example.com --max-pages 50

  # Full workflow
  python main.py ingest --url https://example.com --max-pages 50
  python main.py build-index
  python main.py api  # In one terminal
  python main.py ui   # In another terminal
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build FAISS index")
    build_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing documents",
    )

    # API command
    subparsers.add_parser("api", help="Run FastAPI server")

    # UI command
    subparsers.add_parser("ui", help="Run Streamlit UI")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from URL")
    ingest_parser.add_argument("--url", type=str, required=True, help="URL to crawl")
    ingest_parser.add_argument(
        "--max-pages", type=int, default=50, help="Maximum pages to crawl"
    )
    ingest_parser.add_argument(
        "--output", type=str, default="data/raw", help="Output directory"
    )

    args = parser.parse_args()

    if args.command == "build-index":
        build_index(args.data_dir)
    elif args.command == "api":
        run_api()
    elif args.command == "ui":
        run_ui()
    elif args.command == "ingest":
        ingest_documents(args.url, args.max_pages, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
