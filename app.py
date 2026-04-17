# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from my_agent.agent import graph

load_dotenv()

app = FastAPI(
    title="SOTA RAG Agent API Gateway",
    description="REST API Gateway for the LangGraph RAG pipeline with Pattern A Query Condensation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    success: bool
    raw_query: str
    condensed_query: str
    answer: str
    retrieved_chunks: List[dict]

async def condense_query(query: str, chat_history: List[ChatMessage]) -> str:
    """Pattern A: rewrites follow-up queries into standalone search-optimized questions."""
    if not chat_history:
        return query
    condenser_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history_text = "".join(f"{msg.role.upper()}: {msg.content}\n" for msg in chat_history)
    prompt = f"""Given the following conversation history and a follow-up query, rewrite the follow-up into a standalone, search-optimized question.
Resolve any pronouns based strictly on the chat history. Output only the rewritten question.

Conversation History:
{history_text}
Follow-up Query: {query}
Standalone Question:"""
    try:
        response = await condenser_llm.ainvoke([HumanMessage(content=prompt)])
        condensed = response.content.strip()
        print(f"[Condenser] Raw: \"{query}\" -> Condensed: \"{condensed}\"")
        return condensed
    except Exception as e:
        print(f"[Condenser Error] Fallback: {e}")
        return query

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_agent(request: QueryRequest):
    condensed_search_query = await condense_query(request.query, request.chat_history)
    initial_state = {
        "query": condensed_search_query,
        "retrieved_chunks": [],
        "llm_inputs": [],
        "messages": [],
        "response": None,
        "retry_count": 0
    }
    try:
        final_state = await graph.ainvoke(initial_state)
        answer = final_state.get("response", "Could not generate an answer.")
        chunks = [{"page_content": c["content"], "metadata": c["metadata"]} for c in final_state.get("retrieved_chunks", [])]
        return QueryResponse(
            success=True,
            raw_query=request.query,
            condensed_query=condensed_search_query,
            answer=answer,
            retrieved_chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent RAG workflow: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)