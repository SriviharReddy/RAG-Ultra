import pytest
from my_agent.utils.nodes import merge_chunks_rrf
from my_agent.utils.state import DocumentChunk


def make_chunk(content: str, source: str = "doc.pdf", page: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return {
        "content": content,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        }
    }


def test_basic_merge():
    list_a = [
        make_chunk("Content A1", chunk_index=0),
        make_chunk("Content A2", chunk_index=1),
    ]
    list_b = [
        make_chunk("Content B1", chunk_index=2),
        make_chunk("Content B2", chunk_index=3),
    ]

    merged = merge_chunks_rrf(list_a, list_b, top_n=3)
    assert len(merged) == 3
    # Check that scores are present and positive
    for item in merged:
        assert "score" in item
        assert item["score"] > 0.0


def test_deduplication():
    # chunk1 appears in both lists with identical metadata and content
    chunk_shared = make_chunk("Shared content", chunk_index=0)
    chunk_a = make_chunk("Unique to A", chunk_index=1)
    chunk_b = make_chunk("Unique to B", chunk_index=2)

    list_a = [chunk_shared, chunk_a]
    list_b = [dict(chunk_shared), chunk_b]

    k = 60
    merged = merge_chunks_rrf(list_a, list_b, k=k, top_n=5)

    # Shared chunk should only appear once
    contents = [c["content"] for c in merged]
    assert contents.count("Shared content") == 1
    assert len(merged) == 3

    # Shared chunk was at rank 0 in both lists, so score should be 1/(k+1) + 1/(k+1)
    shared_result = next(c for c in merged if c["content"] == "Shared content")
    expected_score = (1.0 / (k + 1)) + (1.0 / (k + 1))
    assert pytest.approx(shared_result["score"], rel=1e-5) == expected_score


def test_ranking():
    k = 60
    chunk1 = make_chunk("High priority", chunk_index=0)
    chunk2 = make_chunk("Medium priority", chunk_index=1)
    chunk3 = make_chunk("Low priority", chunk_index=2)

    # list_a has chunk1 at rank 0, chunk2 at rank 1
    # list_b has chunk1 at rank 0, chunk3 at rank 1
    # chunk1 appears in both lists at rank 0 -> score: 2 * (1 / (k + 1))
    # chunk2 appears once at rank 1 -> score: 1 / (k + 2)
    # chunk3 appears once at rank 1 -> score: 1 / (k + 2)
    list_a = [chunk1, chunk2]
    list_b = [dict(chunk1), chunk3]

    merged = merge_chunks_rrf(list_a, list_b, k=k, top_n=3)

    assert merged[0]["content"] == "High priority"
    assert merged[0]["score"] > merged[1]["score"]
    assert pytest.approx(merged[1]["score"]) == merged[2]["score"]


def test_empty_inputs():
    chunk1 = make_chunk("Only new content", chunk_index=0)
    new_chunks = [chunk1]

    # Empty existing + non-empty new returns new chunks with scores
    merged_new_only = merge_chunks_rrf([], new_chunks, top_n=4)
    assert len(merged_new_only) == 1
    assert merged_new_only[0]["content"] == "Only new content"
    assert merged_new_only[0]["score"] > 0.0

    # Non-empty existing + empty new returns existing chunks with scores
    existing_chunks = [make_chunk("Only existing content", chunk_index=0)]
    merged_existing_only = merge_chunks_rrf(existing_chunks, [], top_n=4)
    assert len(merged_existing_only) == 1
    assert merged_existing_only[0]["content"] == "Only existing content"
    assert merged_existing_only[0]["score"] > 0.0

    # Both empty returns empty
    merged_empty = merge_chunks_rrf([], [], top_n=4)
    assert merged_empty == []


def test_top_n_clipping():
    chunks_a = [make_chunk(f"A_{i}", chunk_index=i) for i in range(5)]
    chunks_b = [make_chunk(f"B_{i}", chunk_index=i + 10) for i in range(5)]

    # Total 10 distinct chunks, verify clipping to various top_n values
    for top_n in [1, 2, 4, 7]:
        merged = merge_chunks_rrf(chunks_a, chunks_b, top_n=top_n)
        assert len(merged) == top_n

    # top_n larger than total chunks returns all available chunks
    merged_all = merge_chunks_rrf(chunks_a, chunks_b, top_n=20)
    assert len(merged_all) == 10
