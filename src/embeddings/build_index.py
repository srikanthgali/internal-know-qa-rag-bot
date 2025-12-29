import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from tqdm import tqdm

from src.embeddings.embedder import Embedder
from src.ingestion.document_loader import DocumentLoader, Document
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class IndexBuilder:
    """Builds and manages FAISS vector index."""

    def __init__(self):
        """Initialize index builder."""
        self.embedder = Embedder()
        self.doc_loader = DocumentLoader(
            chunk_size=settings.document_processing.chunk_size,
            chunk_overlap=settings.document_processing.chunk_overlap,
            separators=settings.document_processing.separators,
        )
        self.dimension = settings.vector_store.dimension
        self.index_path = settings.vector_store.index_path
        self.metadata_path = settings.vector_store.metadata_path

        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized IndexBuilder")

    def build_index_from_directory(self, data_dir: str) -> None:
        """
        Build FAISS index from documents in directory.

        Args:
            data_dir: Directory containing documents to index
        """
        logger.info(f"Building index from directory: {data_dir}")

        # Load documents
        documents = self.doc_loader.load_directory(data_dir, recursive=True)

        if not documents:
            logger.warning("No documents found to index")
            return

        logger.info(f"Loaded {len(documents)} document chunks")

        # Build index
        self.build_index(documents)

    def build_index(self, documents: List[Document]) -> None:
        """
        Build FAISS index from documents.

        Args:
            documents: List of Document objects to index
        """
        logger.info(f"Building FAISS index for {len(documents)} documents")

        # Enhance text for embedding with metadata
        texts_for_embedding = []
        for doc in documents:
            metadata = doc.metadata
            metadata_prefix = f"Document: {metadata.get('filename', '')} | "
            enhanced_text = metadata_prefix + doc.content
            texts_for_embedding.append(enhanced_text)

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_batch(texts_for_embedding, batch_size=50)

        # Convert to numpy array and normalize
        embeddings_array = np.array(embeddings).astype("float32")

        # FIXED: Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # FIXED: Create FAISS index using Inner Product (cosine similarity)
        logger.info("Creating FAISS index...")
        index = faiss.IndexFlatIP(self.dimension)  # Changed from IndexFlatL2
        index.add(embeddings_array)

        # Prepare metadata
        metadata = []
        for i, doc in enumerate(documents):
            metadata.append(
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "index": i,
                }
            )

        # Save index and metadata
        self._save_index(index, metadata)

        logger.info(f"✓ Index built successfully with {index.ntotal} vectors")

    def _save_index(self, index: faiss.Index, metadata: List[Dict]) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            index: FAISS index object
            metadata: List of document metadata
        """
        try:
            # Save FAISS index
            index_file = self.index_path / "index.faiss"
            faiss.write_index(index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")

            # Save metadata
            metadata_file = self.index_path / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_file}")

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def load_index(self) -> tuple:
        """
        Load FAISS index and metadata from disk.

        Returns:
            Tuple of (index, metadata)
        """
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.json"

            if not index_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(
                    f"Index files not found at {self.index_path}. "
                    "Please build the index first."
                )

            # Load FAISS index
            index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")

            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} documents")

            return index, metadata

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise


def main():
    """Main function to build index from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing documents to index",
    )

    args = parser.parse_args()

    builder = IndexBuilder()
    builder.build_index_from_directory(args.data_dir)


if __name__ == "__main__":
    main()
