import asyncio
from typing import List, Optional, Tuple, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.config import get_settings, get_embeddings

class SotaRagDatabase:
    """
    Persistent, thread-safe Chroma vector database manager.
    Supports single-database parent payload indexing, metadata filtering,
    and non-blocking asynchronous operations via asyncio.to_thread.
    """
    _instance: Optional["SotaRagDatabase"] = None

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.persist_dir
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = get_embeddings()
        self.vector_db = Chroma(
            collection_name=self.collection_name,
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
        metadata_origin: Dict[str, Any]
    ) -> List[str]:
        """
        Ingests child chunks with contextual prefixes. Stores the full parent
        Markdown and image URI directly in each child's metadata payload.
        """
        documents_to_insert = []
        for idx, chunk in enumerate(child_chunks):
            enriched_content = f"[Context: {context_prefix}]\n{chunk}"
            metadata = {
                "parent_content": parent_text,
                "image_url": image_url or "",
                "has_visuals": bool(has_visuals),
                "chunk_index": idx,
                **metadata_origin
            }
            doc = Document(page_content=enriched_content, metadata=metadata)
            documents_to_insert.append(doc)
        
        if documents_to_insert:
            return self.vector_db.add_documents(documents_to_insert)
        return []

    async def ingest_hierarchical_document_async(
        self,
        parent_text: str,
        child_chunks: List[str],
        context_prefix: str,
        image_url: Optional[str],
        has_visuals: bool,
        metadata_origin: Dict[str, Any]
    ) -> List[str]:
        """Async wrapper for non-blocking ingestion."""
        return await asyncio.to_thread(
            self.ingest_hierarchical_document,
            parent_text=parent_text,
            child_chunks=child_chunks,
            context_prefix=context_prefix,
            image_url=image_url,
            has_visuals=has_visuals,
            metadata_origin=metadata_origin
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Synchronous similarity search with relevance scores and optional metadata filter."""
        kwargs: Dict[str, Any] = {"query": query, "k": k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return self.vector_db.similarity_search_with_score(**kwargs)

    async def similarity_search_with_score_async(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Asynchronous similarity search preventing event-loop stalls."""
        return await asyncio.to_thread(
            self.similarity_search_with_score,
            query=query,
            k=k,
            metadata_filter=metadata_filter
        )

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Synchronous similarity search returning documents."""
        kwargs: Dict[str, Any] = {"query": query, "k": k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return self.vector_db.similarity_search(**kwargs)

    async def similarity_search_async(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Asynchronous similarity search."""
        return await asyncio.to_thread(
            self.similarity_search,
            query=query,
            k=k,
            metadata_filter=metadata_filter
        )

    def get_collection_count(self) -> int:
        """Returns the total number of indexed chunk records."""
        try:
            return self.vector_db._collection.count()
        except Exception:
            return 0

    async def get_collection_count_async(self) -> int:
        """Asynchronous retrieval of collection count."""
        return await asyncio.to_thread(self.get_collection_count)


def get_database() -> SotaRagDatabase:
    """Provides a singleton instance of the vector database wrapper."""
    if SotaRagDatabase._instance is None:
        SotaRagDatabase._instance = SotaRagDatabase()
    return SotaRagDatabase._instance
