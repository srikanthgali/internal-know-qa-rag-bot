import numpy as np
from typing import List, Dict, Optional
import faiss

from src.embeddings.embedder import Embedder
from src.embeddings.build_index import IndexBuilder
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class Retriever:
    """Handles document retrieval from FAISS vector store."""

    def __init__(self):
        """Initialize retriever."""
        self.embedder = Embedder()
        self.index_builder = IndexBuilder()
        self.top_k = settings.retrieval.top_k
        self.similarity_threshold = settings.retrieval.similarity_threshold

        # Load index
        self.index, self.metadata = self._load_index()

        logger.info(f"Initialized Retriever with {len(self.metadata)} documents")

    def _load_index(self) -> tuple:
        """Load FAISS index and metadata."""
        try:
            return self.index_builder.load_index()
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info(
                "Please run: python -m src.embeddings.build_index --data-dir data/raw"
            )
            raise

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides config)
            similarity_threshold: Minimum similarity score (overrides config)

        Returns:
            List of retrieved documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        logger.info(f"Retrieving top-{top_k} documents for query: {query[:100]}...")

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            query_vector = np.array([query_embedding]).astype("float32")

            # Search in FAISS index
            distances, indices = self.index.search(query_vector, top_k)

            # Convert distances to similarity scores (cosine similarity)
            # FAISS L2 distance -> similarity score
            similarities = 1 / (1 + distances[0])

            # Prepare results
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if idx < len(self.metadata):
                    doc = self.metadata[idx]
                    results.append(
                        {
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "doc_id": doc["doc_id"],
                            "score": float(similarity),
                        }
                    )

            # Filter by similarity threshold
            results = [r for r in results if r["score"] >= similarity_threshold]

            logger.info(
                f"Retrieved {len(results)} documents (threshold: {similarity_threshold})"
            )

            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    def retrieve_with_metadata_filter(
        self, query: str, metadata_filter: Dict, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents with metadata filtering.

        Args:
            query: Search query
            metadata_filter: Dictionary of metadata filters (e.g., {"file_type": ".pdf"})
            top_k: Number of documents to retrieve

        Returns:
            Filtered list of retrieved documents
        """
        # Get initial results
        results = self.retrieve(
            query, top_k=top_k * 2
        )  # Get more to account for filtering

        # Apply metadata filters
        filtered_results = []
        for result in results:
            metadata = result["metadata"]
            if all(metadata.get(k) == v for k, v in metadata_filter.items()):
                filtered_results.append(result)

        # Limit to top_k
        if top_k:
            filtered_results = filtered_results[:top_k]

        logger.info(f"Filtered to {len(filtered_results)} documents matching criteria")

        return filtered_results
