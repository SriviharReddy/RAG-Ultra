from typing import Any, Literal, TypedDict


class DocumentChunk(TypedDict, total=False):
    content: str
    metadata: dict[str, Any]
    score: float | None

class Citation(TypedDict, total=False):
    id: int
    source: str
    page: int | None
    doc_id: str | None
    snippet: str
    image_url: str | None

class AgentState(TypedDict, total=False):
    query: str
    raw_query: str
    condensed_query: str
    retrieved_chunks: list[DocumentChunk]
    route_decision: Literal["retrieve", "assemble", "generate", "verify", "end"]
    retry_count: int
    critique: str | None
    expanded_query: str | None
    is_relevant: bool | None
    llm_inputs: list[Any]
    answer: str | None
    citations: list[Citation]
    is_grounded: bool | None
    groundedness_score: float | None
    metadata_filter: dict[str, Any] | None


def make_initial_state(
    query: str,
    condensed_query: str,
    metadata_filter: dict[str, Any] | None = None
) -> AgentState:
    return AgentState(
        raw_query=query,
        query=condensed_query,
        condensed_query=condensed_query,
        retrieved_chunks=[],
        route_decision="retrieve",
        retry_count=0,
        critique=None,
        expanded_query=None,
        is_relevant=None,
        llm_inputs=[],
        answer=None,
        citations=[],
        is_grounded=None,
        groundedness_score=None,
        metadata_filter=metadata_filter,
    )
