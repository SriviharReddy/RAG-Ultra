"""
Configuration settings for the RAG-Ultra system
"""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class AWSConfig(BaseModel):
    """AWS Configuration"""
    region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"


class ChromaConfig(BaseModel):
    """ChromaDB Configuration"""
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    collection_name: str = "rag_ultra_docs"


class RetrievalConfig(BaseModel):
    """Retrieval Configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    parent_chunk_size: int = 4000
    top_k_initial: int = 20  # Initial retrieval count
    top_k_final: int = 5     # After reranking


class ModelConfig(BaseModel):
    """Model Configuration"""
    max_tokens: int = 4096
    temperature: float = 0.1
    streaming: bool = True


class Config(BaseModel):
    """Main Configuration"""
    aws: AWSConfig = AWSConfig()
    chroma: ChromaConfig = ChromaConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    model: ModelConfig = ModelConfig()
    
    # Data directories
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    processed_dir: Path = Path("./data/processed")
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [self.data_dir, self.upload_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self


# Global configuration instance
config = Config().ensure_directories()
