from typing import TypedDict, List, Optional, Any

class DocumentChunk(TypedDict):
    content: str
    metadata: dict

class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[DocumentChunk]
    llm_inputs: List[Any]
    messages: List[Any]
    response: Optional[str]
    retry_count: int