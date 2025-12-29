from typing import List, Dict, Optional
from src.config.prompts import (
    QUERY_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE,
    format_context,
    format_chat_history,
)
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class PromptBuilder:
    """Builds prompts for LLM from retrieved context."""

    def __init__(self):
        """
        Initialize prompt builder.

        Args:
            max_context_length: Maximum length of context in characters
        """
        self.max_context_length = settings.generation.max_context_length
        logger.info("Initialized PromptBuilder")

    def build_prompt(
        self,
        question: str,
        retrieved_docs: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build prompt from question and retrieved documents.

        Args:
            question: User question
            retrieved_docs: List of retrieved documents
            chat_history: Optional chat history

        Returns:
            Formatted prompt string
        """
        # Format context from retrieved documents
        context = format_context(retrieved_docs)

        # Truncate context if too long
        if len(context) > self.max_context_length:
            logger.warning(
                f"Context truncated from {len(context)} to {self.max_context_length} chars"
            )
            context = context[: self.max_context_length] + "\n...(truncated)"

        # Build prompt based on whether chat history exists
        if chat_history:
            history_str = format_chat_history(chat_history)
            prompt = CHAT_PROMPT_TEMPLATE.format(
                context=context, chat_history=history_str, question=question
            )
        else:
            prompt = QUERY_PROMPT_TEMPLATE.format(context=context, question=question)

        logger.debug(f"Built prompt with {len(context)} chars of context")

        return prompt

    def build_simple_prompt(self, question: str, context: str) -> str:
        """
        Build simple prompt without chat history.

        Args:
            question: User question
            context: Context string

        Returns:
            Formatted prompt
        """
        return QUERY_PROMPT_TEMPLATE.format(context=context, question=question)

    def _format_context_enhanced(self, retrieved_docs: List[Dict]) -> str:
        """
        Enhanced context formatting with better structure.

        Improvements:
        - Number sources clearly
        - Include relevance scores
        - Add metadata
        - Highlight key information
        """
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("filename", "Unknown")
            content = doc.get("content", "")
            score = doc.get("score", 0.0)

            # Clean and structure content
            content = content.strip()

            # Add structured source block
            source_block = f"""
    ===== SOURCE {i} =====
    File: {source}
    Relevance: {score:.2%}
    ---
    {content}
    ==================
    """
            context_parts.append(source_block)

        return "\n".join(context_parts)
