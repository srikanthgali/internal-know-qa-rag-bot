"""
RAG Evaluation Metrics

Provides comprehensive metrics for evaluating RAG system performance.
"""

from typing import Dict, List, Any, Set
import re
import logging

logger = logging.getLogger(__name__)

# Use NLTK for stop words
try:
    from nltk.corpus import stopwords
    import nltk

    # Download stopwords if not already present
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
except ImportError:
    # Fallback to minimal stop words if NLTK not installed
    logger.warning(
        "NLTK not installed. Using basic stop words. Install with: pip install nltk"
    )
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "can",
        "could",
        "may",
        "might",
        "this",
        "that",
        "these",
        "those",
    }


class RAGMetrics:
    """Metrics calculator for RAG system evaluation."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.stop_words = STOP_WORDS

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        expected_keywords: List[str] = None,
        expected_topics: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a RAG query response.

        Args:
            question: User question
            answer: Generated answer
            retrieved_docs: List of retrieved documents
            expected_keywords: Expected keywords in answer (optional)
            expected_topics: Expected topics in answer (optional)

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # 1. Retrieval Score - quality of retrieved documents
        metrics["retrieval_score"] = self._calculate_retrieval_score(retrieved_docs)

        # 2. Faithfulness - answer grounded in context
        metrics["faithfulness"] = self._calculate_faithfulness(answer, retrieved_docs)

        # 3. Relevance - answer relevance to question
        metrics["relevance"] = self._calculate_relevance(question, answer)

        # 4. Completeness - answer completeness
        metrics["completeness"] = self._calculate_completeness(
            answer, expected_keywords, expected_topics
        )

        # 5. Overall score - weighted average
        metrics["overall"] = self._calculate_overall_score(metrics)

        return metrics

    def _calculate_retrieval_score(self, retrieved_docs: List[Dict]) -> float:
        """
        Calculate retrieval quality score based on document relevance scores.

        Args:
            retrieved_docs: List of retrieved documents with scores

        Returns:
            Retrieval score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0

        # Get raw scores from FAISS (cosine similarity: 0-1 range)
        scores = [doc.get("score", 0.0) for doc in retrieved_docs]

        # BALANCED: Realistic normalization for cosine similarity
        # Maps 0.70-0.95 range to 0.70-0.98 (maintains relative differences)
        normalized_scores = []
        for score in scores:
            if score >= 0.90:  # Excellent match (top tier)
                # 0.90-1.0 → 0.93-0.98
                normalized_scores.append(0.93 + (score - 0.90) * 0.5)
            elif score >= 0.85:  # Very good match
                # 0.85-0.90 → 0.88-0.93
                normalized_scores.append(0.88 + (score - 0.85))
            elif score >= 0.80:  # Good match
                # 0.80-0.85 → 0.82-0.88
                normalized_scores.append(0.82 + (score - 0.80) * 1.2)
            elif score >= 0.75:  # Acceptable match
                # 0.75-0.80 → 0.75-0.82
                normalized_scores.append(0.75 + (score - 0.75) * 1.4)
            elif score >= 0.70:  # Marginal match
                # 0.70-0.75 → 0.65-0.75
                normalized_scores.append(0.65 + (score - 0.70) * 2.0)
            else:  # Below 0.70 is poor
                normalized_scores.append(score * 0.9)  # Apply penalty

        return sum(normalized_scores) / len(normalized_scores)

    def _extract_meaningful_words(self, text: str) -> Set[str]:
        """
        Extract meaningful words from text (excluding stop words).

        Args:
            text: Input text

        Returns:
            Set of meaningful words
        """
        words = re.findall(r"\b\w+\b", text.lower())
        return {word for word in words if len(word) > 3 and word not in self.stop_words}

    def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict]) -> float:
        """
        Calculate faithfulness - whether answer is grounded in retrieved context.

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Faithfulness score between 0 and 1
        """
        if not answer or not retrieved_docs:
            return 0.0

        # Combine all retrieved content
        context = " ".join([doc.get("content", "") for doc in retrieved_docs])

        # Check for "no information" responses - CORRECT behavior
        no_info_phrases = [
            "don't have enough information",
            "couldn't find any relevant information",
            "cannot answer",
            "no information available",
            "not found in",
            "i'm sorry, i couldn't find",
        ]

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in no_info_phrases):
            retrieval_score = self._calculate_retrieval_score(retrieved_docs)
            if not retrieved_docs or retrieval_score < 0.3:
                return 1.0  # Correct handling
            else:
                return 0.8  # Has docs but conservatively declined

        # Split answer into sentences
        answer_sentences = self._split_into_sentences(answer)
        if not answer_sentences:
            return 0.5

        grounded_count = 0
        weakly_grounded_count = 0

        # Extract meaningful words from context once
        context_meaningful_words = self._extract_meaningful_words(context)

        for sentence in answer_sentences:
            # Skip very short transitional sentences
            if len(sentence.split()) < 5:
                grounded_count += 1
                continue

            # Extract meaningful words from sentence
            sentence_meaningful_words = self._extract_meaningful_words(sentence)

            if sentence_meaningful_words:
                # Calculate overlap with context
                overlap = len(
                    sentence_meaningful_words & context_meaningful_words
                ) / len(sentence_meaningful_words)

                # STRICTER THRESHOLDS - Realistic evaluation:
                if overlap >= 0.55:  # Very strong grounding (55%+ overlap)
                    grounded_count += 1.0
                elif overlap >= 0.45:  # Strong grounding (45-55% overlap)
                    grounded_count += 0.85
                elif overlap >= 0.35:  # Moderate grounding (35-45% overlap)
                    grounded_count += 0.65
                elif overlap >= 0.25:  # Weak grounding (25-35% overlap)
                    weakly_grounded_count += 1
                    grounded_count += 0.35
                # else: < 25% overlap = 0 points (likely hallucination or too much synthesis)

        # Calculate base faithfulness score
        total_sentences = len(answer_sentences)
        faithfulness_score = (
            grounded_count / total_sentences if total_sentences > 0 else 0.0
        )

        # STRICTER PENALTY: If many weakly grounded sentences
        weak_ratio = (
            weakly_grounded_count / total_sentences if total_sentences > 0 else 0
        )
        if weak_ratio > 0.25:  # More than 25% weakly grounded (reduced from 30%)
            penalty = 0.15  # Increased penalty from 10% to 15%
            faithfulness_score *= 1 - penalty

        # REMOVED: Structure bonus - it was inflating scores
        # Long answers shouldn't automatically get bonus points

        return faithfulness_score

    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate answer relevance to question."""
        if not answer or not question:
            return 0.0

        # Check for "no information" responses
        no_info_phrases = [
            "don't have enough information",
            "couldn't find any relevant information",
            "cannot answer",
            "no information available",
        ]

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in no_info_phrases):
            return 1.0

        # Extract meaningful question terms
        question_words = re.findall(r"\b\w+\b", question.lower())
        question_stop_words = self.stop_words | {
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "does",
            "gitlab",
            "why",
            "whose",
            "whom",
        }

        question_terms = {
            word
            for word in question_words
            if len(word) > 3 and word not in question_stop_words
        }

        if not question_terms:
            return 0.6

        # Check term coverage
        answer_meaningful_words = self._extract_meaningful_words(answer)
        addressed_terms = len(question_terms & answer_meaningful_words)
        term_coverage = addressed_terms / len(question_terms)

        # Length factor
        word_count = len(answer.split())
        if word_count >= 200:
            length_factor = 0.18  # Reduced from 0.25
        elif word_count >= 100:
            length_factor = 0.15  # Reduced from 0.20
        elif word_count >= 50:
            length_factor = 0.10  # Reduced from 0.12
        else:
            length_factor = 0.05

        # Calculate relevance
        relevance_score = (term_coverage * 0.78) + length_factor  # Reduced weight

        # CAP at 95%
        return min(relevance_score, 0.95)

    def _calculate_completeness(
        self,
        answer: str,
        expected_keywords: List[str] = None,
        expected_topics: List[str] = None,
    ) -> float:
        """Calculate answer completeness."""
        if not answer:
            return 0.0

        no_info_phrases = [
            "don't have enough information",
            "couldn't find any relevant information",
            "cannot answer",
            "no information available",
        ]

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in no_info_phrases):
            return 1.0

        scores = []

        # BALANCED: Moderate structure bonus
        has_numbered_list = bool(re.search(r"\n\d+\.|\n\*\*\d+\.", answer))
        has_bullet_points = answer.count("\n-") >= 3 or answer.count("\n*") >= 3
        has_sections = answer.count("\n\n") >= 2

        structure_bonus = 0.0
        if has_numbered_list:
            structure_bonus = 0.12  # Reduced from 0.20
        elif has_bullet_points:
            structure_bonus = 0.08  # Reduced from 0.15
        elif has_sections:
            structure_bonus = 0.05  # Reduced from 0.08

        # Check keyword coverage
        if expected_keywords:
            keyword_count = sum(
                1 for keyword in expected_keywords if keyword.lower() in answer_lower
            )
            keyword_score = keyword_count / len(expected_keywords)
            if keyword_score < 0.6:
                keyword_score *= 0.85
            scores.append(keyword_score)

        # Check topic coverage
        if expected_topics:
            topic_count = sum(
                1 for topic in expected_topics if topic.lower() in answer_lower
            )
            topic_score = topic_count / len(expected_topics)
            if topic_score < 0.65:
                topic_score *= 0.90
            scores.append(topic_score)

        # Length-based completeness
        word_count = len(answer.split())
        if word_count >= 300:
            length_score = 0.95  # Reduced from 1.0
        elif word_count >= 200:
            length_score = 0.88  # Reduced from 0.90
        elif word_count >= 100:
            length_score = 0.78  # Reduced from 0.75
        elif word_count >= 50:
            length_score = 0.60  # Reduced from 0.55
        else:
            length_score = 0.40

        scores.append(length_score)

        # Calculate final score
        base_score = sum(scores) / len(scores) if scores else 0.5
        final_score = min(base_score + structure_bonus, 0.95)  # Cap at 95%

        return final_score

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted overall score.

        Args:
            metrics: Dictionary of individual metric scores

        Returns:
            Overall score between 0 and 1
        """
        weights = {
            "retrieval_score": 0.30,
            "faithfulness": 0.25,
            "relevance": 0.25,
            "completeness": 0.20,
        }

        weighted_sum = sum(
            metrics.get(metric, 0) * weight for metric, weight in weights.items()
        )

        return weighted_sum

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]
