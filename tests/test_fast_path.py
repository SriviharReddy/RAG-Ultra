import pytest

from rag_pipeline.utils.nodes import evaluate_relevance_node
from rag_pipeline.utils.state import make_initial_state


@pytest.mark.asyncio
async def test_fast_path_bypasses_judge_on_low_distance():
    state = make_initial_state("query", "query")
    state["retrieved_chunks"] = [
        {"content": "High relevance chunk", "metadata": {"source": "manual.pdf", "page": 1}, "score": 0.15}
    ]

    result = await evaluate_relevance_node(state)
    assert result["is_relevant"] is True
    assert result["route_decision"] == "assemble"
    assert "Low distance match" in result["critique"]
    assert result["expanded_query"] is None


@pytest.mark.asyncio
async def test_fast_path_exact_threshold():
    # Exactly 0.3 should still trigger the fast-path
    state = make_initial_state("query", "query")
    state["retrieved_chunks"] = [
        {"content": "Borderline chunk", "metadata": {"source": "manual.pdf", "page": 1}, "score": 0.30}
    ]

    result = await evaluate_relevance_node(state)
    assert result["is_relevant"] is True
    assert result["route_decision"] == "assemble"
    assert "Low distance match" in result["critique"]


@pytest.mark.asyncio
async def test_empty_chunks_routes_to_retry():
    state = make_initial_state("query", "query")
    state["retrieved_chunks"] = []
    state["retry_count"] = 0

    result = await evaluate_relevance_node(state)
    assert result["is_relevant"] is False
    assert result["route_decision"] == "retrieve"
    assert result["retry_count"] == 1
    assert result["expanded_query"] is not None


@pytest.mark.asyncio
async def test_empty_chunks_max_retries_routes_to_assemble():
    state = make_initial_state("query", "query")
    state["retrieved_chunks"] = []
    state["retry_count"] = 3  # max_retries default is 3

    result = await evaluate_relevance_node(state)
    assert result["is_relevant"] is False
    assert result["route_decision"] == "assemble"
