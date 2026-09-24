from rag_pipeline.utils.state import make_initial_state
from schemas import (
    ChatMessage,
    CitationResponse,
    ExecutionMetadata,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


def test_make_initial_state_defaults():
    state = make_initial_state("how to fix turbine?", "how to fix turbine?")
    assert state["raw_query"] == "how to fix turbine?"
    assert state["query"] == "how to fix turbine?"
    assert state["condensed_query"] == "how to fix turbine?"
    assert state["retrieved_chunks"] == []
    assert state["route_decision"] == "retrieve"
    assert state["retry_count"] == 0
    assert state["critique"] is None
    assert state["expanded_query"] is None
    assert state["is_relevant"] is None
    assert state["llm_inputs"] == []
    assert state["answer"] is None
    assert state["citations"] == []
    assert state["is_grounded"] is None
    assert state["groundedness_score"] is None
    assert state["metadata_filter"] is None


def test_make_initial_state_with_filter():
    state = make_initial_state("query", "condensed", {"doc_id": "xyz"})
    assert state["metadata_filter"] == {"doc_id": "xyz"}
    assert state["raw_query"] == "query"
    assert state["condensed_query"] == "condensed"


def test_schemas_roundtrip():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"

    req = QueryRequest(query="test", chat_history=[msg], metadata_filter={"doc_id": "123"})
    assert len(req.chat_history) == 1

    cit = CitationResponse(id=1, source="doc.pdf", page=2, snippet="Sample text")
    assert cit.id == 1
    assert cit.page == 2

    meta = ExecutionMetadata(retry_count=0, latency_ms=123.45, is_relevant=True, is_grounded=True)
    assert meta.latency_ms == 123.45

    res = QueryResponse(
        success=True,
        raw_query="test",
        condensed_query="test",
        answer="The answer",
        citations=[cit],
        retrieved_chunks=[],
        metadata=meta,
    )
    data = res.model_dump()
    assert data["success"] is True
    assert data["citations"][0]["id"] == 1

    ingest = IngestResponse(
        success=True,
        doc_id="doc_1",
        source="doc.pdf",
        pages_processed=1,
        total_chunks_indexed=3,
        message="OK",
    )
    assert ingest.pages_processed == 1

    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        collection_name="test_col",
        collection_count=10,
        models={"fast_llm": "gpt-4o-mini"},
    )
    assert health.status == "healthy"
