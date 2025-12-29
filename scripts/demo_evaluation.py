"""
RAG System Evaluation Script

Evaluates the RAG chatbot on predefined test questions and generates metrics.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.evaluation.metrics.rag_metrics import RAGMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_questions(test_file: str = "tests/test_questions.json") -> List[Dict]:
    """Load test questions from JSON file."""
    test_path = Path(__file__).parent.parent / test_file

    if not test_path.exists():
        raise FileNotFoundError(f"Test questions file not found: {test_path}")

    with open(test_path, "r") as f:
        data = json.load(f)

    return data.get("test_questions", [])


def evaluate_query(
    pipeline: RAGPipeline,
    metrics: RAGMetrics,
    question: str,
    expected_keywords: List[str] = None,
    expected_topics: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single query.

    Args:
        pipeline: RAG pipeline instance
        metrics: Metrics evaluator
        question: User question
        expected_keywords: Expected keywords in answer (optional)
        expected_topics: Expected topics in answer (optional)

    Returns:
        Evaluation results dictionary
    """
    try:
        # Get answer from RAG pipeline
        result = pipeline.query(question, top_k=5)

        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        is_greeting = result.get("is_greeting", False)
        is_intro = result.get("is_intro", False)

        # ADDED: Special handling for greetings/intros
        if is_greeting or is_intro:
            # Greetings should have specific content
            eval_metrics = {
                "retrieval_score": 1.0,  # No retrieval needed
                "faithfulness": 1.0,  # Pre-defined response
                "relevance": 1.0,  # Directly addresses greeting
                "completeness": 1.0,  # Complete greeting response
                "overall": 1.0,
            }

            logger.info(f"Greeting/intro detected: {question[:50]}...")
        else:
            # Calculate metrics for regular queries
            eval_metrics = metrics.evaluate(
                question=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                expected_keywords=expected_keywords or [],
                expected_topics=expected_topics or [],
            )

        # Compile results
        eval_result = {
            "question": question,
            "answer": answer,
            "metrics": eval_metrics,
            "num_sources": len(retrieved_docs),
            "top_source_score": (
                retrieved_docs[0].get("score", 0.0) if retrieved_docs else 0.0
            ),
            "is_greeting": is_greeting,
            "is_intro": is_intro,
        }

        return eval_result

    except Exception as e:
        logger.error(f"Error evaluating query '{question}': {str(e)}")
        return {
            "question": question,
            "error": str(e),
            "metrics": {"overall": 0.0},
        }


def run_evaluation(output_file: str = "evaluation_report.json") -> Dict[str, Any]:
    """
    Run full evaluation suite.

    Args:
        output_file: Path to save evaluation report

    Returns:
        Evaluation report dictionary
    """
    logger.info("Starting RAG evaluation...")

    # Initialize components
    pipeline = RAGPipeline()
    metrics = RAGMetrics()

    # Load test questions
    test_questions = load_test_questions()
    logger.info(f"Loaded {len(test_questions)} test questions")

    # Evaluate each question
    results = []
    for idx, test_case in enumerate(test_questions, 1):
        question = test_case.get("question")

        # Skip if question is missing
        if not question:
            logger.warning(f"Skipping test case {idx}: missing question")
            continue

        logger.info(f"Evaluating {idx}/{len(test_questions)}: {question[:60]}...")

        result = evaluate_query(
            pipeline=pipeline,
            metrics=metrics,
            question=question,
            expected_keywords=test_case.get("expected_keywords", []),
            expected_topics=test_case.get("expected_topics", []),
        )

        if test_case.get("category") in ["edge_case", "impossible"]:
            if (
                "couldn't find" in result.get("answer", "").lower()
                or "don't have" in result.get("answer", "").lower()
            ):
                # Correct handling of edge case
                result["metrics"]["faithfulness"] = 1.0
                result["metrics"]["relevance"] = 1.0
                result["metrics"]["completeness"] = 1.0
                result["metrics"]["overall"] = (
                    result["metrics"]["retrieval_score"] * 0.25 + 0.75
                )

        results.append(result)

    # Calculate summary statistics
    valid_results = [r for r in results if "error" not in r]

    # Separate greetings/intros from other queries
    greeting_categories = ["greeting", "introduction"]
    edge_case_categories = ["edge_case", "impossible"]

    greeting_results = [
        r
        for r in valid_results
        if test_questions[results.index(r)].get("category") in greeting_categories
    ]

    main_results = [
        r
        for r in valid_results
        if test_questions[results.index(r)].get("category") not in greeting_categories
        and test_questions[results.index(r)].get("category") not in edge_case_categories
    ]

    edge_results = [
        r
        for r in valid_results
        if test_questions[results.index(r)].get("category") in edge_case_categories
    ]

    if not main_results:
        logger.error("No valid results to analyze!")
        return {"error": "No valid evaluation results"}

    # Calculate summary for MAIN queries only
    summary = {
        "avg_retrieval_score": sum(
            r["metrics"].get("retrieval_score", 0) for r in main_results
        )
        / len(main_results),
        "avg_faithfulness": sum(
            r["metrics"].get("faithfulness", 0) for r in main_results
        )
        / len(main_results),
        "avg_relevance": sum(r["metrics"].get("relevance", 0) for r in main_results)
        / len(main_results),
        "avg_completeness": sum(
            r["metrics"].get("completeness", 0) for r in main_results
        )
        / len(main_results),
        "avg_overall_score": sum(r["metrics"].get("overall", 0) for r in main_results)
        / len(main_results),
        "total_queries": len(main_results),
        "edge_case_queries": len(edge_results),
        "greeting_queries": len(greeting_results),
    }

    # ADDED: Separate edge case summary
    if edge_results:
        edge_summary = {
            "edge_case_handling": sum(
                1
                for r in edge_results
                if "couldn't find" in r.get("answer", "").lower()
                or "don't have" in r.get("answer", "").lower()
            )
            / len(edge_results),
            "edge_cases_tested": len(edge_results),
        }
        summary.update(edge_summary)

    # ADDED: Greeting handling summary
    if greeting_results:
        greeting_summary = {
            "greeting_handling": sum(
                1 for r in greeting_results if r.get("is_greeting") or r.get("is_intro")
            )
            / len(greeting_results),
            "greetings_tested": len(greeting_results),
        }
        summary.update(greeting_summary)

    # ADD: Variance analysis for quality insights
    faithfulness_scores = [r["metrics"].get("faithfulness", 0) for r in main_results]
    completeness_scores = [r["metrics"].get("completeness", 0) for r in main_results]
    relevance_scores = [r["metrics"].get("relevance", 0) for r in main_results]

    summary["variance_analysis"] = {
        "faithfulness_std_dev": (
            statistics.stdev(faithfulness_scores) if len(faithfulness_scores) > 1 else 0
        ),
        "faithfulness_min": min(faithfulness_scores),
        "faithfulness_max": max(faithfulness_scores),
        "completeness_std_dev": (
            statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0
        ),
        "completeness_min": min(completeness_scores),
        "completeness_max": max(completeness_scores),
        "relevance_std_dev": (
            statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0
        ),
    }

    # Compile final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "queries": results,
    }

    # Save report
    output_path = Path(__file__).parent.parent / output_file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation complete. Report saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries:      {summary['total_queries']}")
    print(f"Retrieval Score:    {summary['avg_retrieval_score']:.2%}")
    print(f"Faithfulness:       {summary['avg_faithfulness']:.2%}")
    print(f"Relevance:          {summary['avg_relevance']:.2%}")
    print(f"Completeness:       {summary['avg_completeness']:.2%}")
    print(f"Overall Score:      {summary['avg_overall_score']:.2%}")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output file for evaluation report",
    )

    args = parser.parse_args()

    try:
        run_evaluation(output_file=args.output)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)
