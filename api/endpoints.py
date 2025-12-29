from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import time

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter()

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
    logger.info("RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    rag_pipeline = None


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str
    max_sources: Optional[int] = 3


class Source(BaseModel):
    """Source document model."""

    content: str
    source: str
    score: float
    metadata: Optional[Dict] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: List[Source]
    query_time: float


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    rag_pipeline_ready: bool
    timestamp: float


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status of the API and RAG pipeline
    """
    return HealthResponse(
        status="healthy",
        rag_pipeline_ready=rag_pipeline is not None,
        timestamp=time.time(),
    )


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with a question.

    Args:
        request: QueryRequest with question and optional parameters

    Returns:
        QueryResponse with answer, sources, and metadata

    Raises:
        HTTPException: If RAG pipeline is not initialized or query fails
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        start_time = time.time()
        logger.info(f"Processing query: {request.question}")

        # Query the RAG pipeline
        result = rag_pipeline.query(
            question=request.question, top_k=request.max_sources
        )

        query_time = time.time() - start_time

        # Format sources
        sources = [
            Source(
                content=doc.get("content", ""),
                source=doc.get("metadata", {}).get("source", "Unknown"),
                score=float(doc.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
            )
            for doc in result.get("retrieved_docs", [])
        ]

        logger.info(f"Query completed in {query_time:.2f}s with {len(sources)} sources")

        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            query_time=query_time,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the knowledge base.

    Returns:
        Statistics including document count and index info
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        stats = {
            "total_documents": (
                len(rag_pipeline.retriever.metadata)
                if hasattr(rag_pipeline, "retriever")
                else 0
            ),
            "index_type": "FAISS",
            "embedding_model": (
                rag_pipeline.retriever.embedder.model
                if hasattr(rag_pipeline, "retriever")
                else "unknown"
            ),
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
