# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from my_agent.agent import graph

load_dotenv()

app = FastAPI(title="RAG-Ultra API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    chunks: list

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_agent(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "retrieved_chunks": [],
        "llm_inputs": [],
        "messages": [],
        "response": None,
        "retry_count": 0
    }
    try:
        final_state = await graph.ainvoke(initial_state)
        return QueryResponse(answer=final_state.get("response", ""), chunks=final_state.get("retrieved_chunks", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)