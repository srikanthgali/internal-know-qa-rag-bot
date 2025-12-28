from typing import List, Dict, Optional
from src.config.prompts import (
    QUERY_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE,
    format_context,
    format_chat_history,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Builds prompts for LLM from retrieved context."""

    def __init__(self, max_context_length: int = 3000):
        """
        Initialize prompt builder.

        Args:
            max_context_length: Maximum length of context in characters
        """
        self.max_context_length = max_context_length
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
