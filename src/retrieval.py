"""
Retrieval Module with Parent Document Retriever, ChromaDB, and AWS-based Reranking
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import boto3
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

from src.config import config


class BedrockEmbeddings(Embeddings):
    """
    Amazon Bedrock Titan Embeddings wrapper for LangChain
    """
    
    def __init__(
        self, 
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = None
    ):
        self.model_id = model_id
        self.region = region or config.aws.region
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.region
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string"""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using Titan"""
        # Truncate if too long (Titan has 8k token limit)
        text = text[:25000]  # Approximate character limit
        
        request_body = {
            "inputText": text,
            "dimensions": 1024,  # Titan v2 supports 256, 512, 1024
            "normalize": True
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['embedding']


class BedrockReranker:
    """
    Reranker using Amazon Bedrock Cohere Rerank or LLM-based reranking
    Uses Cohere Rerank through Bedrock when available, falls back to LLM scoring
    """
    
    def __init__(self, model_id: str = "cohere.rerank-v3-5:0"):
        self.model_id = model_id
        self.region = config.aws.region
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.region
        )
        self._use_cohere = self._check_cohere_available()
    
    def _check_cohere_available(self) -> bool:
        """Check if Cohere rerank is available in the region"""
        try:
            bedrock = boto3.client('bedrock', region_name=self.region)
            models = bedrock.list_foundation_models()
            for model in models.get('modelSummaries', []):
                if 'rerank' in model.get('modelId', '').lower():
                    return True
            return False
        except Exception:
            return False
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on relevance to query
        """
        if not documents:
            return []
        
        if self._use_cohere:
            return self._rerank_with_cohere(query, documents, top_k)
        else:
            return self._rerank_with_llm(query, documents, top_k)
    
    def _rerank_with_cohere(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """Use Cohere Rerank through Bedrock"""
        doc_texts = [doc.page_content for doc in documents]
        
        request_body = {
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(documents)),
            "return_documents": False
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        results = response_body.get('results', [])
        
        reranked = []
        for result in results:
            idx = result['index']
            score = result['relevance_score']
            reranked.append((documents[idx], score))
        
        return reranked
    
    def _rerank_with_llm(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        LLM-based reranking using Amazon Nova Lite for scoring relevance
        """
        scored_docs = []
        
        for doc in documents:
            score = self._score_document(query, doc.page_content)
            scored_docs.append((doc, score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def _score_document(self, query: str, content: str) -> float:
        """Score a single document's relevance to query using LLM"""
        # Truncate content if too long
        content = content[:3000]
        
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0-100.
Only respond with a single integer number, nothing else.

Query: {query}

Document: {content}

Relevance score (0-100):"""

        request_body = {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": 10,
                "temperature": 0.0
            }
        }
        
        try:
            response = self.client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                body=json.dumps(request_body),
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            score_text = response_body['output']['message']['content'][0]['text'].strip()
            
            # Parse score
            score = int(''.join(filter(str.isdigit, score_text[:5])))
            return min(100, max(0, score)) / 100.0
        except Exception:
            return 0.5  # Default mid-score on error


class VectorStoreManager:
    """
    Manages ChromaDB vector store with parent document retrieval
    """
    
    def __init__(self):
        self.embeddings = BedrockEmbeddings()
        self.reranker = BedrockReranker()
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Parent document storage (using in-memory for child-parent mapping)
        self.parent_store = InMemoryStore()
        
        # Text splitters for child/parent chunking
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.retrieval.parent_chunk_size,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize or load vector store
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        """Initialize the Chroma vector store"""
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=config.chroma.collection_name,
            embedding_function=self.embeddings
        )
        
        # Create parent document retriever
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.parent_store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": config.retrieval.top_k_initial}
        )
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents using parent document retrieval strategy
        Returns count of documents added
        """
        # Use parent document retriever to add docs
        self.parent_retriever.add_documents(documents)
        return len(documents)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None
    ) -> List[Document]:
        """
        Retrieve relevant documents with reranking
        """
        top_k = top_k or config.retrieval.top_k_final
        
        # Initial retrieval (gets parent documents)
        initial_docs = self.parent_retriever.invoke(query)
        
        if not initial_docs:
            return []
        
        # Rerank and filter
        reranked = self.reranker.rerank(
            query, 
            initial_docs, 
            top_k=top_k
        )
        
        # Return documents with scores in metadata
        result_docs = []
        for doc, score in reranked:
            doc.metadata['relevance_score'] = score
            result_docs.append(doc)
        
        return result_docs
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Simple similarity search without parent retrieval
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)"""
        self.chroma_client.delete_collection(config.chroma.collection_name)
        self._init_vectorstore()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        collection = self.chroma_client.get_collection(config.chroma.collection_name)
        return {
            "name": config.chroma.collection_name,
            "count": collection.count(),
            "metadata": collection.metadata
        }


class HybridRetriever:
    """
    Hybrid retriever combining semantic search with structured retrieval
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_manager = vector_store_manager
    
    def retrieve(
        self, 
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_tables: bool = True,
        top_k: int = 5
    ) -> List[Document]:
        """
        Hybrid retrieval with optional metadata filtering
        """
        # Get all relevant documents
        docs = self.vector_manager.retrieve(query, top_k=top_k * 2)
        
        # Apply metadata filters
        if filter_metadata:
            docs = [
                doc for doc in docs
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        
        # Optionally filter out or prioritize table content
        if not include_tables:
            docs = [doc for doc in docs if doc.metadata.get('doc_type') != 'table']
        
        return docs[:top_k]
    
    def retrieve_with_context(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve documents with additional context about the retrieval
        """
        docs = self.vector_manager.retrieve(query, top_k=top_k)
        
        # Aggregate metadata
        sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
        pages = list(set(doc.metadata.get('page', 0) for doc in docs))
        has_tables = any(doc.metadata.get('doc_type') == 'table' for doc in docs)
        
        return {
            "documents": docs,
            "sources": sources,
            "pages": sorted(pages),
            "has_tables": has_tables,
            "total_retrieved": len(docs)
        }
