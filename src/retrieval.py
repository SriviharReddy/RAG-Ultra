from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config

class RetrievalEngine:
    def __init__(self):
        self.embedding = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # In-memory retrieval setup for the demo; in production use persistent clients
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Parent Document Retrieval architecture
        self.docstore = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def add_documents(self, text: str, metadata: dict = None):
        """
        Ingests raw text into the parent-child retriever.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        self.retriever.add_documents([doc])

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """
        Standard retrieval.
        """
        return self.retriever.invoke(query_text)
    
    def rerank(self, documents: List[Document], query_text: str) -> List[Document]:
        """
        Placeholder for Cross-Encoder reranking (e.g. Cohere or BAAI/bge-reranker).
        """
        # Mock ranking logic: assumes all are relevant, just returns them.
        # integration: 
        # scores = cross_encoder.predict([(query_text, doc.page_content) for doc in documents])
        # then sort by scores.
        return documents
