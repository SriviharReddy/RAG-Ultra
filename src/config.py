import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    CHROMA_PERSIST_DIRECTORY = "chroma_db"
    COLLECTION_NAME = "rag_ultra_docs"
    
    # Models
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    VLM_MODEL = "gpt-4o"  # Using GPT-4o for visual reasoning as well

config = Config()
