from typing import TypedDict, List, Optional, Dict, Any, Literal

class DocumentChunk(TypedDict, total=False):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float]

class Citation(TypedDict, total=False):
    id: int
    source: str
    page: Optional[int]
    doc_id: Optional[str]
    snippet: str
    image_url: Optional[str]

class AgentState(TypedDict, total=False):
    query: str
    raw_query: str
    condensed_query: str
    retrieved_chunks: List[DocumentChunk]
    route_decision: Literal["retrieve", "assemble", "generate", "verify", "end"]
    retry_count: int
    critique: Optional[str]
    expanded_query: Optional[str]
    is_relevant: Optional[bool]
    llm_inputs: List[Any]
    answer: Optional[str]
    citations: List[Citation]
    is_grounded: Optional[bool]
    groundedness_score: Optional[float]
    metadata_filter: Optional[Dict[str, Any]]
