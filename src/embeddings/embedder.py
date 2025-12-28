import os
from typing import List, Union
import numpy as np
from openai import OpenAI
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class Embedder:
    """Handles text embedding generation using OpenAI."""

    def __init__(self):
        """Initialize embedder with OpenAI client."""
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.embedding_model
        self.dimension = settings.vector_store.dimension
        logger.info(f"Initialized Embedder with model: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()

            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension

            response = self.client.embeddings.create(input=text, model=self.model)

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Clean texts
                batch = [text.replace("\n", " ").strip() for text in batch]

                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )

                response = self.client.embeddings.create(input=batch, model=self.model)

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        logger.debug(f"Generating query embedding for: {query[:100]}...")
        return self.embed_text(query)
