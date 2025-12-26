#!/usr/bin/env python3
"""
Internal Knowledge Base RAG Chatbot Project Structure Generator

A streamlined, portfolio-grade project structure for building a production-ready
RAG chatbot in one week. Focuses on core functionality while maintaining
professional standards for interview demonstrations.

Features:
    - Document ingestion (PDF, TXT, DOCX, .MD, etc.)
    - Embedding generation with OpenAI
    - Vector storage with FAISS
    - Simple retrieval logic
    - FastAPI REST API
    - Streamlit chat interface
    - Configuration via YAML
    - Basic logging and testing

Author: Srikanth Gali
"""

import logging
from pathlib import Path
from typing import Dict
from datetime import datetime


class RAGChatbotProjectGenerator:
    """
    Generates a RAG chatbot project structure for Internal Knowledge Base.

    This generator creates a minimal project layout that is easy to understand,
    and extend. It focuses on essential components needed
    for a functional RAG chatbot while maintaining clean code organization.

    Attributes:
        project_name (str): Name of the project
        base_path (Path): Base directory path where project will be created
        created_count (int): Counter for tracking created files/directories

    Example:
        >>> generator = RAGChatbotProjectGenerator()
        >>> generator.create_project_structure()
    """

    def __init__(self, project_name: str = "internal-know-qa-rag-bot"):
        """
        Initialize the project generator.

        Args:
            project_name (str): Name of the project
        """
        self.project_name = project_name
        self.base_path = Path(".")
        self.created_count = 0

    def create_project_structure(self) -> None:
        """
        Create the complete project directory structure.

        Creates a streamlined structure focusing on:
        - Core RAG functionality
        - Clean code organization
        - Easy to understand and extend
        - Documentation
        """
        logging.info(f"Creating project structure for {self.project_name}")
        logging.info(f"Base path: {self.base_path.absolute()}")

        self._create_directories()
        self._create_files()
        self._create_gitkeep_files()

        logging.info(
            f"✓ {self.project_name} structure created successfully at: {self.base_path.absolute()}"
        )
        logging.info(f"✓ Total new files and directories created: {self.created_count}")

    def _create_directories(self) -> None:
        """
        Create essential project directories.

        Minimal structure focused on core functionality.
        """
        directories = [
            # Data
            "data/raw",
            "data/processed",
            # Core source code
            "src/config",
            "src/ingestion",
            "src/embeddings",
            "src/retrieval",
            "src/generation",
            "src/utils",
            # API
            "api",
            # Streamlit UI
            "ui/streamlit_app/components",
            "ui/streamlit_app/styles",
            # Storage
            "vector_store",
            "vector_store/faiss_index",
            # Logs
            "logs",
            # Tests (basic)
            "tests",
            # Documentation
            "docs",
            # Notebooks for exploration and evaluation
            "notebooks",
        ]

        for directory in directories:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.created_count += 1
                logging.info(f"✓ Created: {directory}")

    def _create_files(self) -> None:
        """
        Create essential project files.

        Focused on must-have files for a working demo and interview presentation.
        """
        files = [
            # Root configuration
            ".gitignore",
            ".env.example",
            "README.md",
            "requirements.txt",
            "config.yaml",
            # Source - Config
            "src/__init__.py",
            "src/config/__init__.py",
            "src/config/settings.py",
            "src/config/logging_config.yaml",
            "src/config/prompts.py",
            # Source - Data Ingestion
            "src/ingestion/__init__.py",
            "src/ingestion/document_loader.py",
            "src/ingestion/text_processor.py",
            # Source - Embeddings
            "src/embeddings/__init__.py",
            "src/embeddings/embedder.py",
            "src/embeddings/build_index.py",
            # Source - Retrieval
            "src/retrieval/__init__.py",
            "src/retrieval/retriever.py",
            "src/retrieval/rag_pipeline.py",
            # Source - Generation
            "src/generation/__init__.py",
            "src/generation/llm.py",
            "src/generation/prompt_builder.py",
            # Source - Utils
            "src/utils/__init__.py",
            "src/utils/logger.py",
            "src/utils/helpers.py",
            # API
            "api/__init__.py",
            "api/main.py",
            "api/models.py",
            "api/endpoints.py",
            # Streamlit UI
            "ui/streamlit_app/__init__.py",
            "ui/streamlit_app/app.py",
            "ui/streamlit_app/components/__init__.py",
            "ui/streamlit_app/components/chat.py",
            "ui/streamlit_app/components/sidebar.py",
            "ui/streamlit_app/styles/custom.css",
            # Scripts
            "scripts/ingest_documents.py",
            # Tests
            "tests/__init__.py",
            "tests/test_retrieval.py",
            "tests/test_generation.py",
            # Documentation
            "docs/SETUP.md",
            "docs/API_DOCS.md",
            "docs/ARCHITECTURE.md",
            # Notebooks
            "notebooks/exploration.ipynb",
            "notebooks/evaluation.ipynb",
            # Main entry points
            "main.py",
            "run_api.py",
            "run_streamlit.py",
        ]

        for file_path_str in files:
            file_path = self.base_path / file_path_str
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.touch()
                self.created_count += 1
                logging.info(f"✓ Created: {file_path_str}")

    def _create_gitkeep_files(self) -> None:
        """Create .gitkeep files in empty directories."""
        gitkeep_dirs = [
            "data/raw",
            "data/processed",
            "vector_store",
            "logs",
        ]

        for gitkeep_dir in gitkeep_dirs:
            gitkeep_path = self.base_path / gitkeep_dir / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
                self.created_count += 1
                logging.info(f"✓ Created: {gitkeep_dir}/.gitkeep")

    def generate_summary_report(self) -> Dict[str, any]:
        """
        Generate a summary report of the project structure.

        Returns:
            Dict containing project statistics and metadata
        """
        return {
            "project_name": self.project_name,
            "base_path": str(self.base_path.absolute()),
            "new_items_created": self.created_count,
            "timestamp": datetime.now().isoformat(),
            "status": "Internal KnowledgeBase RAG chatbot structure",
            "estimated_completion": "1 week",
            "key_features": [
                "Document ingestion (PDF, TXT, DOCX, MD, etc.)",
                "Vector search with FAISS",
                "OpenAI embeddings",
                "FastAPI REST API",
                "Streamlit chat interface",
                "Configuration management",
                "Basic logging and testing",
            ],
        }


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the project generator.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    setup_logging(log_level="INFO")

    generator = RAGChatbotProjectGenerator()

    try:
        generator.create_project_structure()

        summary = generator.generate_summary_report()
        print("\n" + "=" * 70)
        print("🚀 INTERNAL KNOWLEDGEBASE RAG CHATBOT - PROJECT STRUCTURE")
        print("=" * 70)
        print(f"Project Name: {summary['project_name']}")
        print(f"Location: {summary['base_path']}")
        print(f"Items Created: {summary['new_items_created']}")
        print(f"Status: {summary['status']}")
        print(f"Estimated Time: {summary['estimated_completion']}")
        print("\n📋 Key Features:")
        for feature in summary["key_features"]:
            print(f"   ✓ {feature}")
        print("\n💡 Next Steps:")
        print("   1. Update .env.example with your API keys")
        print("   2. Install requirements: pip install -r requirements.txt")
        print("   3. Add documents to data/raw/")
        print("   4. Run: python scripts/ingest_documents.py")
        print("   5. Start API: python run_api.py")
        print("   6. Launch UI: python run_streamlit.py")
        print("=" * 70 + "\n")

    except Exception as e:
        logging.error(f"Error creating project structure: {e}")
        raise
