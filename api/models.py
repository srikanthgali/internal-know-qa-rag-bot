from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for question answering."""

    question: str = Field(..., description="User question", min_length=1)
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous chat messages"
    )
    top_k: Optional[int] = Field(
        default=5, description="Number of documents to retrieve", ge=1, le=20
    )


class Source(BaseModel):
    """Source document information."""

    filename: str
    source: str
    file_type: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for question answering."""

    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents")
    model: str = Field(..., description="Model used for generation")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    index_loaded: bool
    document_count: int
