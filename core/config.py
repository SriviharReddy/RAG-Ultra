import logging
import os
import zlib
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Version tag for the offline embedding algorithm. Stored as collection
# metadata so legacy vectors (created under hash() or MD5) are detected
# and rebuilt during database initialization.
EMBEDDING_VERSION = "crc32-v1"

class DeterministicOfflineEmbeddings(Embeddings):
    """
    Zero-crash offline embedding model for testing and demonstration when no OpenAI API key is supplied.
    Produces deterministic, normalized term-hashed vectors that maintain semantic keyword overlap.
    """
    def __init__(self, size: int = 1536):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.size, dtype=np.float32)
        words = text.lower().replace("\n", " ").replace("|", " ").replace("-", " ").split()
        for w in words:
            if w:
                h = zlib.crc32(w.encode("utf-8")) % self.size
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Endpoints
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # OCR Provider Settings (Novita AI, OpenAI Vision, or custom VLM endpoint)
    ocr_provider: str = "auto"  # "auto", "openai", "novita", "custom"
    ocr_model: str = "gpt-4o-mini"
    ocr_api_key: str | None = None
    ocr_base_url: str | None = None
    novita_api_key: str | None = None
    novita_base_url: str = "https://api.novita.ai/v1"
    novita_model: str = "qwen/qwen-2.5-vl-72b-instruct"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # Model Provider Settings
    fast_llm_model: str = "gpt-4o-mini"
    generation_llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Storage Paths
    persist_dir: str = "./db_storage/chroma"
    image_storage_dir: str = "./db_storage/images"
    collection_name: str = "sota_rag_collection"
    embedding_version: str = EMBEDDING_VERSION

    # Agentic Execution Controls
    max_retries: int = 3
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 3

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    uvicorn_reload: bool = False

    def ensure_directories(self) -> None:
        """Ensures that required storage directories exist on disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.image_storage_dir, exist_ok=True)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

_llm_cache: dict[tuple, ChatOpenAI] = {}

def get_fast_llm(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Returns the low-latency LLM for routing, evaluation, query condensation, and grading."""
    settings = get_settings()
    cache_key = ("fast", settings.fast_llm_model, temperature)
    if cache_key in _llm_cache and not kwargs:
        return _llm_cache[cache_key]

    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key configured; LLM calls will fail unless using a keyless base_url.")
        api_key = "not-set"

    llm_kwargs = {
        "model": settings.fast_llm_model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url

    llm = ChatOpenAI(**llm_kwargs)
    if not kwargs:
        _llm_cache[cache_key] = llm
    return llm

def get_generation_llm(temperature: float = 0.1, **kwargs) -> ChatOpenAI:
    """Returns the flagship multimodal LLM for final generation and synthesis."""
    settings = get_settings()
    cache_key = ("generation", settings.generation_llm_model, temperature)
    if cache_key in _llm_cache and not kwargs:
        return _llm_cache[cache_key]

    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key configured; LLM calls will fail unless using a keyless base_url.")
        api_key = "not-set"

    llm_kwargs = {
        "model": settings.generation_llm_model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url

    llm = ChatOpenAI(**llm_kwargs)
    if not kwargs:
        _llm_cache[cache_key] = llm
    return llm

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Returns the embedding model instance."""
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if api_key:
        emb_kwargs = {
            "model": settings.embedding_model,
            "api_key": api_key
        }
        if settings.openai_base_url:
            emb_kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**emb_kwargs)

    # Graceful fallback to deterministic embeddings
    return DeterministicOfflineEmbeddings()
