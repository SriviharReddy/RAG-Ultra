import os
from functools import lru_cache
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

class DeterministicOfflineEmbeddings(Embeddings):
    """
    Zero-crash offline embedding model for testing and demonstration when no OpenAI API key is supplied.
    Produces deterministic, normalized term-hashed vectors that maintain semantic keyword overlap.
    """
    def __init__(self, size: int = 1536):
        self.size = size

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.size, dtype=np.float32)
        words = text.lower().replace("\n", " ").replace("|", " ").replace("-", " ").split()
        for w in words:
            if w:
                h = abs(hash(w)) % self.size
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Endpoints
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    novita_api_key: Optional[str] = None

    # Model Provider Settings
    fast_llm_model: str = "gpt-4o-mini"
    generation_llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Storage Paths
    persist_dir: str = "./db_storage/chroma"
    image_storage_dir: str = "./db_storage/images"
    collection_name: str = "sota_rag_collection"

    # Agentic Execution Controls
    max_retries: int = 3
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 3

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080

    def ensure_directories(self) -> None:
        """Ensures that required storage directories exist on disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.image_storage_dir, exist_ok=True)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

def get_fast_llm(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Returns the low-latency LLM for routing, evaluation, query condensation, and grading."""
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "sk-dummy-key"
    llm_kwargs = {
        "model": settings.fast_llm_model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**llm_kwargs)

def get_generation_llm(temperature: float = 0.1, **kwargs) -> ChatOpenAI:
    """Returns the flagship multimodal LLM for final generation and synthesis."""
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "sk-dummy-key"
    llm_kwargs = {
        "model": settings.generation_llm_model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**llm_kwargs)

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
