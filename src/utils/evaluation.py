"""
Evaluation utilities for RAG chatbot.

Provides metrics and tools for assessing retrieval and generation quality.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json
from src.retrieval.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    question: str
    answer: str
    expected_answer: Optional[str]
    retrieved_docs: List[Dict]

    # Retrieval metrics
    retrieval_score: float
    top_doc_relevance: float

    # Answer metrics
    faithfulness_score: float
    relevance_score: float
    completeness_score: float

    # Overall
    overall_score: float
    notes: str = ""


class RAGEvaluator:
    """Evaluates RAG chatbot performance."""

    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize evaluator.

        Args:
            pipeline: RAGPipeline instance to evaluate
        """
        self.pipeline = pipeline
        logger.info("Initialized RAG Evaluator")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict],
        expected_keywords: Optional[List[str]] = None,
    ) -> float:
        """
        Evaluate retrieval quality.

        Args:
            query: Search query
            retrieved_docs: Retrieved documents
            expected_keywords: Optional keywords that should appear

        Returns:
            Retrieval quality score (0-1)
        """
        if not retrieved_docs:
            return 0.0

        score = 0.0

        # Check average relevance score
        avg_relevance = np.mean([doc.get("score", 0) for doc in retrieved_docs])
        score += avg_relevance * 0.4

        # Check if expected keywords are present
        if expected_keywords:
            content = " ".join([doc.get("content", "") for doc in retrieved_docs])
            keyword_matches = sum(
                1 for kw in expected_keywords if kw.lower() in content.lower()
            )
            keyword_score = keyword_matches / len(expected_keywords)
            score += keyword_score * 0.6
        else:
            # If no keywords, use relevance score more heavily
            score += avg_relevance * 0.6

        return min(score, 1.0)

    def evaluate_faithfulness(self, answer: str, retrieved_docs: List[Dict]) -> float:
        """
        Evaluate if answer is faithful to retrieved context.

        Uses simple heuristic: checks if key phrases from answer
        appear in retrieved documents.

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Faithfulness score (0-1)
        """
        if not retrieved_docs or not answer:
            return 0.0

        # Extract key phrases (simple approach: split on sentences)
        answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

        if not answer_sentences:
            return 0.5  # Neutral score for very short answers

        # Combine all retrieved content
        context = " ".join([doc.get("content", "") for doc in retrieved_docs])
        context_lower = context.lower()

        # Check how many answer phrases appear in context
        faithful_count = 0
        for sentence in answer_sentences:
            # Extract important words (simple: words longer than 4 chars)
            important_words = [w.lower() for w in sentence.split() if len(w) > 4]

            if not important_words:
                continue

            # Check if majority of important words appear in context
            matches = sum(1 for word in important_words if word in context_lower)
            if matches / len(important_words) >= 0.6:
                faithful_count += 1

        faithfulness = faithful_count / len(answer_sentences) if answer_sentences else 0
        return faithfulness

    def evaluate_relevance(
        self, question: str, answer: str, expected_topics: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate answer relevance to question.

        Args:
            question: User question
            answer: Generated answer
            expected_topics: Optional topics that should be covered

        Returns:
            Relevance score (0-1)
        """
        if not answer or answer.startswith("I don't have enough information"):
            return 0.0

        score = 0.5  # Base score for non-empty answer

        # Check if answer contains question keywords
        question_words = set(w.lower() for w in question.split() if len(w) > 4)
        answer_words = set(w.lower() for w in answer.split())

        keyword_overlap = len(question_words & answer_words)
        if question_words:
            score += (keyword_overlap / len(question_words)) * 0.3

        # Check for expected topics
        if expected_topics:
            answer_lower = answer.lower()
            topic_matches = sum(
                1 for topic in expected_topics if topic.lower() in answer_lower
            )
            score += (topic_matches / len(expected_topics)) * 0.2

        return min(score, 1.0)

    def evaluate_completeness(self, answer: str, min_length: int = 50) -> float:
        """
        Evaluate answer completeness.

        Args:
            answer: Generated answer
            min_length: Minimum expected answer length

        Returns:
            Completeness score (0-1)
        """
        if not answer:
            return 0.0

        # Length-based score
        length_score = min(len(answer) / min_length, 1.0) * 0.5

        # Structure score (has multiple sentences)
        sentences = [s for s in answer.split(".") if len(s.strip()) > 10]
        structure_score = min(len(sentences) / 3, 1.0) * 0.5

        return length_score + structure_score

    def evaluate_query(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
        expected_topics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of a single query.

        Args:
            question: User question
            expected_answer: Optional expected answer for comparison
            expected_keywords: Keywords that should appear in retrieved docs
            expected_topics: Topics that should be covered in answer

        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating query: {question}")

        # Get response from pipeline
        result = self.pipeline.query(question)
        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])

        # Calculate metrics
        retrieval_score = self.evaluate_retrieval(
            question, retrieved_docs, expected_keywords
        )

        top_doc_relevance = retrieved_docs[0].get("score", 0) if retrieved_docs else 0

        faithfulness_score = self.evaluate_faithfulness(answer, retrieved_docs)

        relevance_score = self.evaluate_relevance(question, answer, expected_topics)

        completeness_score = self.evaluate_completeness(answer)

        # Overall score (weighted average)
        overall_score = (
            retrieval_score * 0.3
            + faithfulness_score * 0.3
            + relevance_score * 0.2
            + completeness_score * 0.2
        )

        return EvaluationResult(
            question=question,
            answer=answer,
            expected_answer=expected_answer,
            retrieved_docs=retrieved_docs,
            retrieval_score=retrieval_score,
            top_doc_relevance=top_doc_relevance,
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
        )

    def evaluate_test_set(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test case dicts with 'question' and optional
                       'expected_keywords', 'expected_topics'

        Returns:
            Dictionary with aggregate results and individual scores
        """
        results = []

        for test_case in test_cases:
            result = self.evaluate_query(
                question=test_case["question"],
                expected_keywords=test_case.get("expected_keywords"),
                expected_topics=test_case.get("expected_topics"),
            )
            results.append(result)

        # Calculate aggregate metrics
        avg_metrics = {
            "avg_retrieval_score": np.mean([r.retrieval_score for r in results]),
            "avg_faithfulness": np.mean([r.faithfulness_score for r in results]),
            "avg_relevance": np.mean([r.relevance_score for r in results]),
            "avg_completeness": np.mean([r.completeness_score for r in results]),
            "avg_overall_score": np.mean([r.overall_score for r in results]),
            "total_queries": len(results),
        }

        logger.info(f"Evaluated {len(results)} queries")
        logger.info(f"Average overall score: {avg_metrics['avg_overall_score']:.3f}")

        return {"aggregate_metrics": avg_metrics, "individual_results": results}


def generate_evaluation_report(
    evaluation_results: Dict, output_file: str = "evaluation_report.json"
) -> None:
    """
    Generate evaluation report.

    Args:
        evaluation_results: Results from evaluate_test_set
        output_file: Output file path
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": evaluation_results["aggregate_metrics"],
        "queries": [],
    }

    for result in evaluation_results["individual_results"]:
        report["queries"].append(
            {
                "question": result.question,
                "answer": result.answer,
                "metrics": {
                    "retrieval": result.retrieval_score,
                    "faithfulness": result.faithfulness_score,
                    "relevance": result.relevance_score,
                    "completeness": result.completeness_score,
                    "overall": result.overall_score,
                },
                "num_sources": len(result.retrieved_docs),
                "top_source_score": result.top_doc_relevance,
            }
        )

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()
    evaluator = RAGEvaluator(pipeline)

    # Sample test cases
    test_cases = [
        {
            "question": "What is the code review process?",
            "expected_keywords": ["review", "code", "approval"],
            "expected_topics": ["code review", "process", "workflow"],
        },
        {
            "question": "How do I deploy to production?",
            "expected_keywords": ["deploy", "production", "release"],
            "expected_topics": ["deployment", "production", "steps"],
        },
    ]

    # Run evaluation
    results = evaluator.evaluate_test_set(test_cases)

    # Generate report
    generate_evaluation_report(results)

    # Print summary
    print("\n=== Evaluation Summary ===")
    for metric, value in results["aggregate_metrics"].items():
        print(f"{metric}: {value:.3f}")
