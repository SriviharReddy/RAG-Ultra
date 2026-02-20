import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from my_agent.utils.state import AgentState
from my_agent.utils.tools import vector_search_db

load_dotenv()

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant chunks from vector store."""
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    results = vector_search_db.invoke({"query": query, "k": 3})
    chunks = [{"content": r.get("parent_content", r.get("content", "")), "metadata": {k: v for k, v in r.items() if k not in ("content", "parent_content")}} for r in results]
    return {"retrieved_chunks": chunks, "retry_count": retry_count + 1}

async def evaluate_relevance_node(state: AgentState) -> Dict[str, Any]:
    """LLM-as-Judge: evaluates whether retrieved context is sufficient."""
    query = state["query"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)
    if not chunks:
        return {"response": "generate" if retry_count >= 3 else "retrieve"}
    contexts = [c["content"] for c in chunks]
    prompt = f"""Analyze if the retrieved context is sufficient to answer the query.
Query: {query}
Contexts: {' '.join(contexts)}
Reply ONLY with 'YES' or 'NO'."""
    judge_response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    decision = "generate"
    if "NO" in judge_response.content.strip().upper() and retry_count < 3:
        decision = "retrieve"
    return {"response": decision}

async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Final generation node."""
    query = state["query"]
    context = "\n\n".join([c["content"] for c in state["retrieved_chunks"]])
    prompt = f"Answer strictly based on the context.\n\nQuestion: {query}\n\nContext:\n{context}"
    response = await generation_llm.ainvoke([HumanMessage(content=prompt)])
    return {"response": response.content}