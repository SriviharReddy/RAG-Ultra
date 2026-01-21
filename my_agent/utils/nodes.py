import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from my_agent.utils.state import AgentState
from my_agent.utils.tools import vector_search_db

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant chunks from vector store."""
    query = state["query"]
    results = vector_search_db.invoke({"query": query, "k": 3})
    chunks = [{"content": r["content"], "metadata": {k: v for k, v in r.items() if k != "content"}} for r in results]
    return {"retrieved_chunks": chunks}

async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Generates a response from retrieved context."""
    query = state["query"]
    context = "\n\n".join([c["content"] for c in state["retrieved_chunks"]])
    prompt = f"Answer this question based on the context below.\n\nQuestion: {query}\n\nContext:\n{context}"
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"response": response.content}