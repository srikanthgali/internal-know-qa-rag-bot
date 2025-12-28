import re
import logging
from typing import List, Optional
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Processes and cleans text for better embedding quality."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_extra_whitespace: bool = True,
        lowercase: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 10,
    ):
        """
        Initialize the text processor.

        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses from text
            remove_extra_whitespace: Normalize whitespace
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters (keep alphanumeric and punctuation)
            min_length: Minimum length for processed text chunks
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length

    def process(self, text: str) -> str:
        """
        Process a single text string.

        Args:
            text: Input text to process

        Returns:
            Processed text string
        """
        if not text or not text.strip():
            return ""

        # Normalize unicode characters
        text = self._normalize_unicode(text)

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Remove special characters
        if self.remove_special_chars:
            text = self._remove_special_chars(text)

        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = self._normalize_whitespace(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Final cleanup
        text = text.strip()

        # Check minimum length
        if len(text) < self.min_length:
            logger.debug(f"Text too short after processing: {len(text)} chars")
            return ""

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process multiple text strings.

        Args:
            texts: List of input texts

        Returns:
            List of processed text strings (empty strings removed)
        """
        processed = []
        for text in texts:
            result = self.process(text)
            if result:
                processed.append(result)

        logger.info(
            f"Processed {len(texts)} texts, kept {len(processed)} after filtering"
        )
        return processed

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to ASCII equivalents where possible."""
        # Normalize to NFKD form and encode to ASCII, ignoring errors
        text = unicodedata.normalize("NFKD", text)
        # Keep unicode characters but normalize them
        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, "", text)
        # Also remove www URLs
        www_pattern = r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(www_pattern, "", text)
        return text

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return re.sub(email_pattern, "", text)

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        pattern = r"[^a-zA-Z0-9\s.,!?;:()\-\'\"]"
        return re.sub(pattern, "", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces, tabs, newlines)."""
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        # Remove empty lines and join
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def clean_for_search(self, query: str) -> str:
        """
        Clean a search query text.

        Args:
            query: Search query string

        Returns:
            Cleaned query string
        """
        # More aggressive cleaning for queries
        query = query.strip()
        query = self._normalize_unicode(query)
        query = self._normalize_whitespace(query)

        # Remove special characters but keep basic punctuation
        query = re.sub(r"[^\w\s\-\']", " ", query)
        query = re.sub(r"\s+", " ", query).strip()

        return query


class DocumentDeduplicator:
    """Remove duplicate or near-duplicate document chunks."""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Threshold for considering documents as duplicates (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts based on exact matching.

        Args:
            texts: List of text strings

        Returns:
            List of unique texts
        """
        seen = set()
        unique_texts = []

        for text in texts:
            # Create a normalized version for comparison
            normalized = text.strip().lower()

            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_texts.append(text)

        removed = len(texts) - len(unique_texts)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate texts")

        return unique_texts

    def deduplicate_by_hash(self, texts: List[str]) -> List[str]:
        """
        Remove duplicates using hash-based comparison (faster for large datasets).

        Args:
            texts: List of text strings

        Returns:
            List of unique texts
        """
        seen_hashes = set()
        unique_texts = []

        for text in texts:
            text_hash = hash(text.strip().lower())

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        removed = len(texts) - len(unique_texts)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate texts using hash method")

        return unique_texts


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = TextProcessor(
        remove_urls=True,
        remove_emails=True,
        remove_extra_whitespace=True,
        lowercase=False,
        min_length=10,
    )

    # Example text
    sample_text = """
    Check out our website at https://example.com for more info!
    Contact us at info@example.com


    This is   some    sample    text    with    extra    spaces.
    """

    # Process text
    cleaned = processor.process(sample_text)
    print("Original:")
    print(sample_text)
    print("\nCleaned:")
    print(cleaned)

    # Process batch
    texts = [sample_text, "Short", "Another valid text that is long enough"]
    cleaned_batch = processor.process_batch(texts)
    print(f"\nProcessed batch: {len(cleaned_batch)} texts kept")

    # Deduplicate
    deduplicator = DocumentDeduplicator()
    duplicates = ["same text", "same text", "different text", "same text"]
    unique = deduplicator.deduplicate(duplicates)
    print(f"\nDeduplication: {len(duplicates)} -> {len(unique)} unique texts")
