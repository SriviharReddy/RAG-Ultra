# core/database.py
import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class SotaRagDatabase:
    def __init__(self):
        self.persist_dir = os.getenv("PERSIST_DIR", "./db_storage/chroma")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_db = Chroma(
            collection_name="sota_rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def ingest_document(self, text: str, metadata: dict):
        """Basic single-document ingestion."""
        doc = Document(page_content=text, metadata=metadata)
        self.vector_db.add_documents([doc])
        print(f"Ingested document with metadata: {metadata}")