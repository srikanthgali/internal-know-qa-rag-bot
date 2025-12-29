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
        self.rerank = settings.retrieval.rerank
        self.similarity_threshold = settings.retrieval.similarity_threshold
        self.edge_case_min_score = settings.retrieval.edge_case_min_score

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
        Retrieve most relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of retrieved documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        logger.info(f"Retrieving top-{top_k} documents for query: {query}...")

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

            # Normalize query vector for cosine similarity
            faiss.normalize_L2(query_vector)

            # Search in FAISS index - retrieve MORE to account for deduplication
            search_k = top_k * 3
            distances, indices = self.index.search(query_vector, search_k)

            # Convert to results with metadata
            results = []
            seen_sources = set()

            # ADDED: Track maximum score to detect low-quality retrieval
            max_score = 0.0

            for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
                if doc_idx == -1:
                    break

                # With IndexFlatIP, distances ARE similarity scores (0-1 range)
                similarity_score = float(distance)

                # CAP at 1.0 for safety
                normalized_score = min(max(similarity_score, 0.0), 1.0)

                # Track max score
                if normalized_score > max_score:
                    max_score = normalized_score

                # Get metadata
                if isinstance(self.metadata, dict):
                    doc_metadata = self.metadata.get(str(doc_idx), {})
                elif isinstance(self.metadata, list):
                    doc_metadata = (
                        self.metadata[doc_idx]
                        if 0 <= doc_idx < len(self.metadata)
                        else {}
                    )
                else:
                    doc_metadata = {}

                content = doc_metadata.get("content", "")
                source_file = doc_metadata.get("metadata", {}).get("source", "")

                # Deduplicate by source file
                if source_file in seen_sources:
                    continue
                seen_sources.add(source_file)

                # Apply threshold filter
                if normalized_score >= similarity_threshold:
                    results.append(
                        {
                            "content": content,
                            "metadata": doc_metadata.get("metadata", {}),
                            "score": normalized_score,
                        }
                    )

                # Stop once we have enough unique sources
                if len(results) >= top_k:
                    break

            # FIX: If max score is below edge case min score, consider this a low-quality retrieval
            # Return empty results to trigger "no information" response
            if max_score < self.edge_case_min_score:
                logger.warning(
                    f"Low retrieval quality (max score: {max_score:.2%}). "
                    f"Returning no results for out-of-scope query."
                )
                return []

            logger.info(
                f"Retrieved {len(results)} unique documents (threshold: {similarity_threshold})"
            )

            # Optional: Rerank results
            if self.rerank:
                results = self._rerank_results(query, results)

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

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better retrieval.

        - Expand acronyms
        - Add context keywords
        - Remove noise
        """
        # Remove question marks and lowercase
        query = query.replace("?", "").strip()

        # Add common context (customize for your domain)
        if len(query.split()) < 5:
            query = f"{query} process steps documentation"

        return query

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using keyword matching boost.

        Args:
            query: Search query
            results: Initial retrieval results

        Returns:
            Reranked results with capped scores
        """
        query_terms = set(query.lower().split())

        for result in results:
            content_lower = result["content"].lower()
            keyword_matches = sum(1 for term in query_terms if term in content_lower)

            # FIXED: Add boost instead of multiply, then cap
            boost = min(keyword_matches * 0.05, 0.15)  # Max 15% boost
            result["score"] = min(result["score"] + boost, 1.0)  # Cap at 1.0

        # Re-sort by boosted scores
        return sorted(results, key=lambda x: x["score"], reverse=True)
