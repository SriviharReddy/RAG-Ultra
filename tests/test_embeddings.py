import math

import numpy as np

from core.config import DeterministicOfflineEmbeddings, Settings


def test_determinism():
    embeddings = DeterministicOfflineEmbeddings(size=512)
    text = "Artificial intelligence and machine learning in multi-modal retrieval."
    vec1 = embeddings.embed_query(text)
    vec2 = embeddings.embed_query(text)
    assert vec1 == vec2


def test_normalization():
    embeddings = DeterministicOfflineEmbeddings(size=1024)
    vec = embeddings.embed_query("Query for vector normalization testing.")
    l2_norm = float(np.linalg.norm(vec))
    assert math.isclose(l2_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_uvicorn_reload_parses_boolean_environment_value(monkeypatch):
    monkeypatch.setenv("UVICORN_RELOAD", "true")

    settings = Settings(_env_file=None)

    assert settings.uvicorn_reload is True


def test_dimensionality():
    # Test custom size
    custom_size = 256
    embeddings_custom = DeterministicOfflineEmbeddings(size=custom_size)
    vec_custom = embeddings_custom.embed_query("Sample text")
    assert len(vec_custom) == custom_size

    # Test default size (1536)
    embeddings_default = DeterministicOfflineEmbeddings()
    assert embeddings_default.size == 1536
    vec_default = embeddings_default.embed_query("Sample text")
    assert len(vec_default) == 1536


def test_distinct_inputs():
    embeddings = DeterministicOfflineEmbeddings(size=512)
    vec1 = embeddings.embed_query("Quantum computing and superposition")
    vec2 = embeddings.embed_query("Baking sourdough bread with hydration techniques")
    assert vec1 != vec2


def test_embed_documents():
    embeddings = DeterministicOfflineEmbeddings(size=256)
    docs = [
        "First document content",
        "Second document content",
        "Third document content with different topics",
    ]
    vectors = embeddings.embed_documents(docs)
    assert len(vectors) == len(docs)
    for doc, vec in zip(docs, vectors):
        assert len(vec) == 256
        assert vec == embeddings.embed_query(doc)


def test_empty_and_whitespace_input():
    embeddings = DeterministicOfflineEmbeddings(size=128)
    for text in ["", "   ", "\n\t  \n"]:
        vec = embeddings.embed_query(text)
        assert len(vec) == 128
        assert all(v == 0.0 for v in vec)
        assert np.linalg.norm(vec) == 0.0
