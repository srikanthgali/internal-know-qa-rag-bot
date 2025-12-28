from typing import List, Dict, Optional
from openai import OpenAI

from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(self):
        """Initialize RAG pipeline components."""
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens

        logger.info("Initialized RAG Pipeline")

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User question
            chat_history: Previous chat messages
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing query: {question}")

        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I'm sorry, I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "retrieved_docs": [],
                }

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Step 2: Build prompt
            prompt = self.prompt_builder.build_prompt(
                question=question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history,
            )

            # Step 3: Generate answer using LLM
            logger.info("Generating answer...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": settings.get("generation.system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            # Step 4: Extract sources
            sources = self._extract_sources(retrieved_docs)

            logger.info("Query processed successfully")

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_docs": retrieved_docs,
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Extract and format source information.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()

        for doc in retrieved_docs:
            metadata = doc["metadata"]
            source_id = metadata.get("filename", "Unknown")

            if source_id not in seen_sources:
                sources.append(
                    {
                        "filename": metadata.get("filename", "Unknown"),
                        "source": metadata.get("source", ""),
                        "file_type": metadata.get("file_type", ""),
                        "relevance_score": doc["score"],
                    }
                )
                seen_sources.add(source_id)

        return sources

    def stream_query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Process query with streaming response.

        Args:
            question: User question
            chat_history: Previous chat messages
            top_k: Number of documents to retrieve

        Yields:
            Chunks of generated text
        """
        logger.info(f"Processing streaming query: {question}")

        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            if not retrieved_docs:
                yield "I'm sorry, I couldn't find any relevant information to answer your question."
                return

            # Build prompt
            prompt = self.prompt_builder.build_prompt(
                question=question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history,
            )

            # Stream response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": settings.get("generation.system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"Error: {str(e)}"
