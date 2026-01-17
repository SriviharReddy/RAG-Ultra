"""
LangGraph Workflow for Multimodal RAG Pipeline
Implements the complete RAG flow with decision nodes for retrieval, reranking, and generation
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any, Sequence
from operator import add

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.retrieval import VectorStoreManager, HybridRetriever
from src.generation import ResponseGenerator, QueryAnalyzer
from src.config import config


class RAGState(TypedDict):
    """State for the RAG workflow"""
    # Input
    query: str
    chat_history: List[Dict[str, str]]
    
    # Processing
    query_analysis: Dict[str, Any]
    expanded_queries: List[str]
    
    # Retrieval
    retrieved_documents: List[Document]
    retrieval_context: Dict[str, Any]
    
    # Generation
    response: str
    sources: List[str]
    
    # Metadata
    messages: Annotated[Sequence[BaseMessage], add_messages]
    error: Optional[str]


class RAGWorkflow:
    """
    LangGraph-based RAG workflow with multi-stage retrieval and generation
    """
    
    def __init__(self):
        # Initialize components
        self.vector_manager = VectorStoreManager()
        self.retriever = HybridRetriever(self.vector_manager)
        self.generator = ResponseGenerator()
        self.query_analyzer = QueryAnalyzer()
        
        # Build the workflow graph
        self.graph = self._build_graph()
        
        # Memory for conversation persistence
        self.memory = MemorySaver()
        
        # Compile the graph
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("expand_query", self._expand_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("check_retrieval", self._check_retrieval)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("fallback", self._fallback_response)
        
        # Define edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "check_retrieval")
        
        # Conditional edge based on retrieval results
        workflow.add_conditional_edges(
            "check_retrieval",
            self._should_generate,
            {
                "generate": "generate",
                "fallback": "fallback"
            }
        )
        
        workflow.add_edge("generate", END)
        workflow.add_edge("fallback", END)
        
        return workflow
    
    def _analyze_query(self, state: RAGState) -> Dict[str, Any]:
        """Analyze the user query to determine retrieval strategy"""
        query = state["query"]
        
        analysis = self.query_analyzer.classify_query_type(query)
        
        return {
            "query_analysis": analysis,
            "messages": [HumanMessage(content=query)]
        }
    
    def _expand_query(self, state: RAGState) -> Dict[str, Any]:
        """Expand query into multiple search variants"""
        query = state["query"]
        analysis = state.get("query_analysis", {})
        
        # Only expand for complex queries
        if analysis.get("complexity") == "complex":
            expanded = self.query_analyzer.expand_query(query)
        else:
            expanded = [query]
        
        return {"expanded_queries": expanded}
    
    def _retrieve_documents(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        queries = state.get("expanded_queries", [state["query"]])
        analysis = state.get("query_analysis", {})
        
        # Collect documents from all query variants
        all_docs = []
        seen_content = set()
        
        for query in queries:
            context = self.retriever.retrieve_with_context(
                query,
                top_k=config.retrieval.top_k_initial
            )
            
            for doc in context["documents"]:
                # Deduplicate by content hash
                content_hash = hash(doc.page_content[:500])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        # Filter based on query type
        if analysis.get("needs_tables"):
            # Prioritize table documents
            table_docs = [d for d in all_docs if d.metadata.get("doc_type") == "table"]
            other_docs = [d for d in all_docs if d.metadata.get("doc_type") != "table"]
            all_docs = table_docs + other_docs
        
        # Trim to final count
        final_docs = all_docs[:config.retrieval.top_k_final]
        
        # Build context info
        sources = list(set(d.metadata.get("source", "Unknown") for d in final_docs))
        
        return {
            "retrieved_documents": final_docs,
            "retrieval_context": {
                "total_retrieved": len(final_docs),
                "sources": sources,
                "has_tables": any(d.metadata.get("doc_type") == "table" for d in final_docs)
            }
        }
    
    def _check_retrieval(self, state: RAGState) -> Dict[str, Any]:
        """Check if retrieval was successful"""
        docs = state.get("retrieved_documents", [])
        
        # Check if we have meaningful results
        has_results = len(docs) > 0
        
        # Check relevance scores if available
        if has_results:
            scores = [d.metadata.get("relevance_score", 0.5) for d in docs]
            avg_score = sum(scores) / len(scores)
            has_relevant = avg_score > 0.3
        else:
            has_relevant = False
        
        return {
            "retrieval_context": {
                **state.get("retrieval_context", {}),
                "has_results": has_results,
                "is_relevant": has_relevant
            }
        }
    
    def _should_generate(self, state: RAGState) -> str:
        """Determine whether to generate or fallback"""
        context = state.get("retrieval_context", {})
        
        if context.get("has_results") and context.get("is_relevant", True):
            return "generate"
        return "fallback"
    
    def _generate_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate response using LLM"""
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        chat_history = state.get("chat_history", [])
        
        # Generate response
        response = self.generator.generate(
            query=query,
            context_docs=docs,
            chat_history=chat_history
        )
        
        # Extract sources
        sources = list(set(
            f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 'N/A')})"
            for d in docs
        ))
        
        return {
            "response": response,
            "sources": sources,
            "messages": [AIMessage(content=response)]
        }
    
    def _fallback_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate fallback response when retrieval fails"""
        query = state["query"]
        
        fallback_msg = f"""I couldn't find relevant information in the uploaded documents to answer your question: "{query}"

This could mean:
1. The documents don't contain information about this topic
2. The question might need to be rephrased
3. More relevant documents need to be uploaded

Would you like to:
- Try rephrasing your question?
- Upload additional documents?
- Ask about a different topic from the existing documents?"""

        return {
            "response": fallback_msg,
            "sources": [],
            "messages": [AIMessage(content=fallback_msg)]
        }
    
    def invoke(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Run the RAG workflow for a query
        """
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "messages": [],
            "error": None
        }
        
        config_dict = {"configurable": {"thread_id": session_id}}
        
        result = self.app.invoke(initial_state, config_dict)
        
        return {
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "retrieved_docs": result.get("retrieved_documents", []),
            "query_analysis": result.get("query_analysis", {})
        }
    
    def stream(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default"
    ):
        """
        Stream the RAG workflow execution
        Yields intermediate states and final response
        """
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "messages": [],
            "error": None
        }
        
        config_dict = {"configurable": {"thread_id": session_id}}
        
        for event in self.app.stream(initial_state, config_dict, stream_mode="values"):
            yield event


class DocumentManager:
    """
    Manages document ingestion into the RAG system
    """
    
    def __init__(self, workflow: RAGWorkflow):
        self.workflow = workflow
        self.vector_manager = workflow.vector_manager
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to the vector store"""
        count = self.vector_manager.add_documents(documents)
        stats = self.vector_manager.get_collection_stats()
        
        return {
            "added": count,
            "total_documents": stats.get("count", 0),
            "collection": stats.get("name", "")
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_manager.get_collection_stats()
    
    def clear_all(self):
        """Clear all documents from the store"""
        self.vector_manager.delete_collection()


# Create singleton instances
rag_workflow = RAGWorkflow()
document_manager = DocumentManager(rag_workflow)
