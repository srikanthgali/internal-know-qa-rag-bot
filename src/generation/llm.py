from typing import List, Dict, Optional, Generator
from openai import OpenAI

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class LLM:
    """Language Model wrapper for text generation."""

    def __init__(self):
        """Initialize LLM client."""
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens

        logger.info(f"Initialized LLM with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text
        """
        if system_prompt is None:
            system_prompt = settings.get("generation.system_prompt")

        if temperature is None:
            temperature = self.temperature

        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text with chat history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.temperature

        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating with history: {e}")
            raise

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override

        Yields:
            Text chunks
        """
        if system_prompt is None:
            system_prompt = settings.get("generation.system_prompt")

        if temperature is None:
            temperature = self.temperature

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
