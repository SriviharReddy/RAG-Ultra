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

    def ingest_hierarchical_document(
        self,
        parent_text: str,
        child_chunks: List[str],
        context_prefix: str,
        image_url: Optional[str],
        has_visuals: bool,
        metadata_origin: dict
    ):
        """
        Ingests child chunks with contextual prefixes. Stores the full parent
        Markdown and image URI directly in each child's metadata payload.
        """
        documents_to_insert = []
        for chunk in child_chunks:
            enriched_content = f"[Context: {context_prefix}]\n{chunk}"
            metadata = {
                "parent_content": parent_text,
                "image_url": image_url,
                "has_visuals": has_visuals,
                **metadata_origin
            }
            doc = Document(page_content=enriched_content, metadata=metadata)
            documents_to_insert.append(doc)
        self.vector_db.add_documents(documents_to_insert)