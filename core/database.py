import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import get_embeddings, get_settings

logger = logging.getLogger(__name__)


class ParentStore:
    """Lightweight JSON-backed store for parent page content, avoiding
    duplication across child chunks in Chroma metadata."""

    def __init__(self, persist_dir: str):
        self._path = Path(persist_dir) / "parent_store.json"
        self._data: dict[str, str] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def put(self, doc_id: str, page: int | str, content: str) -> str:
        """Store parent content and return the lookup key."""
        key = f"{doc_id}::{page}"
        self._data[key] = content
        return key

    def get(self, key: str) -> str:
        """Retrieve parent content by key."""
        return self._data.get(key, "")

    def flush(self) -> None:
        """Persist to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def clear_doc(self, doc_id: str) -> None:
        """Remove all entries for a document (used during re-ingestion)."""
        self._data = {k: v for k, v in self._data.items() if not k.startswith(f"{doc_id}::")}

    def clear(self) -> None:
        """Clear all entries and remove persisted file."""
        self._data.clear()
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass

class SotaRagDatabase:
    """
    Persistent, thread-safe Chroma vector database manager.
    Supports single-database parent payload indexing, metadata filtering,
    and non-blocking asynchronous operations via asyncio.to_thread.
    """
    _instance: "SotaRagDatabase | None" = None

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.persist_dir
        self.collection_name = collection_name or settings.collection_name
        self.embedding_version = settings.embedding_version
        self.embeddings = get_embeddings()

        # Detect and rebuild collections persisted under an incompatible
        # embedding algorithm so stale vectors are never queried.
        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self._migrate_legacy_collection()

        self.vector_db = Chroma(
            client=self._chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine", "embedding_version": self.embedding_version}
        )
        self.parent_store = ParentStore(self.persist_dir)

    def _migrate_legacy_collection(self) -> None:
        """Detect collections created with an incompatible embedding version and rebuild them.

        When the offline embedding algorithm changes (e.g. hash() -> MD5 -> CRC32),
        previously persisted vectors become incompatible with new queries. This method
        checks the persisted collection's version marker and destroys it for a clean
        rebuild if the version does not match.
        """
        try:
            coll = self._chroma_client.get_collection(self.collection_name)
            meta = coll.metadata or {}
            if meta.get("embedding_version") != self.embedding_version:
                logger.warning(
                    "Collection '%s' has embedding version '%s'; expected '%s'. "
                    "Rebuilding collection (destructive migration).",
                    self.collection_name,
                    meta.get("embedding_version"),
                    self.embedding_version,
                )
                self._chroma_client.delete_collection(self.collection_name)
                if hasattr(self, "parent_store"):
                    self.parent_store.clear()
                else:
                    store_file = Path(self.persist_dir) / "parent_store.json"
                    if store_file.exists():
                        try:
                            store_file.unlink()
                        except OSError:
                            pass
        except Exception:
            # Collection does not exist yet — created automatically by Chroma
            pass

    def ingest_hierarchical_document(
        self,
        parent_text: str,
        child_chunks: list[str],
        context_prefix: str,
        image_url: str | None,
        has_visuals: bool,
        metadata_origin: dict[str, Any]
    ) -> list[str]:
        """
        Ingests child chunks with contextual prefixes. Stores parent content in
        parent store and associates chunks via parent_key in metadata.
        """
        doc_id = str(metadata_origin.get("doc_id", "doc"))
        page = metadata_origin.get("page", 0)
        parent_key = self.parent_store.put(doc_id, page, parent_text)

        documents_to_insert = []
        for idx, chunk in enumerate(child_chunks):
            enriched_content = f"[Context: {context_prefix}]\n{chunk}"
            metadata = {
                "image_url": image_url or "",
                "has_visuals": bool(has_visuals),
                "chunk_index": idx,
                **metadata_origin,
                "parent_key": parent_key,
            }
            metadata.pop("parent_content", None)
            doc = Document(page_content=enriched_content, metadata=metadata)
            documents_to_insert.append(doc)

        if documents_to_insert:
            res = self.vector_db.add_documents(documents_to_insert)
            self.parent_store.flush()
            return res
        return []

    def get_parent_content(self, key: str) -> str:
        """Retrieve parent content by key from the parent store."""
        return self.parent_store.get(key)

    async def ingest_hierarchical_document_async(
        self,
        parent_text: str,
        child_chunks: list[str],
        context_prefix: str,
        image_url: str | None,
        has_visuals: bool,
        metadata_origin: dict[str, Any]
    ) -> list[str]:
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
        metadata_filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Synchronous similarity search with relevance scores and optional metadata filter."""
        kwargs: dict[str, Any] = {"query": query, "k": k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return self.vector_db.similarity_search_with_score(**kwargs)

    async def similarity_search_with_score_async(
        self,
        query: str,
        k: int = 3,
        metadata_filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
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
        metadata_filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Synchronous similarity search returning documents."""
        kwargs: dict[str, Any] = {"query": query, "k": k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return self.vector_db.similarity_search(**kwargs)

    async def similarity_search_async(
        self,
        query: str,
        k: int = 3,
        metadata_filter: dict[str, Any] | None = None
    ) -> list[Document]:
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


_db_lock = threading.Lock()

def get_database() -> SotaRagDatabase:
    """Provides a singleton instance of the vector database wrapper."""
    if SotaRagDatabase._instance is None:
        with _db_lock:
            if SotaRagDatabase._instance is None:
                SotaRagDatabase._instance = SotaRagDatabase()
    return SotaRagDatabase._instance
