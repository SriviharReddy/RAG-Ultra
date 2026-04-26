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

# Load environment variables
load_dotenv()

app = FastAPI(
    title="SOTA RAG Agent API Gateway",
    description="REST API Gateway exposing the Compiled LangGraph active document-inference workflow with Pattern A Query Condensation.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schema Definitions ---
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
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

# --- Pattern A: Query Condensation Helper ---
async def condense_query(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Evaluates raw query and preceding chat history. If history is present,
    invokes a fast model (gpt-5.5-instant) to rewrite the pronoun-dependent 
    follow-up query into an explicit, search-optimized standalone question.
    """
    if not chat_history:
        return query
        
    # Standard 2026 low-latency model for pre-processing condensation
    condenser_llm = ChatOpenAI(model="gpt-5.5-instant", temperature=0)
    
    # Format conversational history log
    history_text = ""
    for msg in chat_history:
        history_text += f"{msg.role.upper()}: {msg.content}\n"
        
    prompt = f"""
    Given the following conversation history and a follow-up query, rewrite the follow-up query into a single standalone, search-optimized question.
    The standalone question must resolve any ambiguous pronouns (such as "it", "this", "that", "they") based strictly on the chat history.
    Do not answer the question, only output the rewritten standalone question.
    
    Conversation History:
    {history_text}
    
    Follow-up Query: {query}
    
    Standalone Question:
    """
    try:
        response = await condenser_llm.ainvoke([HumanMessage(content=prompt)])
        condensed = response.content.strip()
        print(f"[Query Condenser] Raw: \"{query}\" -> Condensed: \"{condensed}\"")
        return condensed
    except Exception as e:
        print(f"[Query Condenser Error] Fallback to raw query due to: {e}")
        return query

# --- REST Route ---
@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_agent(request: QueryRequest):
    """
    Gateway Endpoint. Receives the query and optional chat history.
    Applies Pattern A query condensation, executes the stateless LangGraph pipeline,
    and returns context-grounded Markdown answers.
    """
    # 1. Condense the query if conversational history exists (Pattern A)
    condensed_search_query = await condense_query(request.query, request.chat_history)
    
    # 2. Setup initial graph state (completely stateless, no checkpointer config required)
    initial_state = {
        "query": condensed_search_query,
        "retrieved_chunks": [],
        "llm_inputs": [],
        "messages": [],
        "response": None,
        "retry_count": 0
    }
    
    try:
        # 3. Run the compiled stateless LangGraph RAG workflow
        final_state = await graph.ainvoke(initial_state)
        
        # 4. Extract final answer and clean retrieved parent contexts
        answer = final_state.get("response", "Could not generate an answer.")
        chunks = []
        
        for c in final_state.get("retrieved_chunks", []):
            chunks.append({
                "page_content": c["content"],
                "metadata": c["metadata"]
            })
            
        return QueryResponse(
            success=True,
            raw_query=request.query,
            condensed_query=condensed_search_query,
            answer=answer,
            retrieved_chunks=chunks
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent RAG workflow: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)