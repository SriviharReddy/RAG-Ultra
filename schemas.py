from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message text")


class QueryRequest(BaseModel):
    query: str = Field(description="User query or follow-up question")
    chat_history: list[ChatMessage] | None = Field(default=[], description="Preceding conversation context")
    metadata_filter: dict[str, Any] | None = Field(default=None, description="Optional Chroma metadata filter (e.g. {'doc_id': 'xyz'})")


class CitationResponse(BaseModel):
    id: int
    source: str
    page: int | None = None
    doc_id: str | None = None
    snippet: str
    image_url: str | None = None


class ExecutionMetadata(BaseModel):
    retry_count: int
    latency_ms: float
    is_relevant: bool | None = None
    is_grounded: bool | None = None
    groundedness_score: float | None = None
    critique: str | None = None


class QueryResponse(BaseModel):
    success: bool
    raw_query: str
    condensed_query: str
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[dict[str, Any]]
    metadata: ExecutionMetadata


class IngestResponse(BaseModel):
    success: bool
    doc_id: str
    source: str
    pages_processed: int
    total_chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    collection_name: str
    collection_count: int
    models: dict[str, str]
