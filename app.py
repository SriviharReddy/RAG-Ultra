import os
import time
import json
import uuid
import tempfile
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.config import get_settings, get_fast_llm
from core.database import get_database
from ingest_cli import ingest_file
from my_agent.agent import graph

load_dotenv()
settings = get_settings()

app = FastAPI(
    title="RAG-Ultra Agent API Gateway",
    description="Microservice exposing Corrective Multimodal RAG (CRAG) with Query Condensation, LLM-as-a-Judge, and SSE Streaming.",
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

# Mount local image storage for static browser and client viewing
settings.ensure_directories()
app.mount("/static/images", StaticFiles(directory=settings.image_storage_dir), name="static_images")

# --- Schema Definitions ---

class ChatMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message text")

class QueryRequest(BaseModel):
    query: str = Field(description="User query or follow-up question")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Preceding conversation context")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional Chroma metadata filter (e.g. {'doc_id': 'xyz'})")

class CitationResponse(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    doc_id: Optional[str] = None
    snippet: str
    image_url: Optional[str] = None

class ExecutionMetadata(BaseModel):
    retry_count: int
    latency_ms: float
    is_relevant: Optional[bool] = None
    is_grounded: Optional[bool] = None
    groundedness_score: Optional[float] = None
    critique: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    raw_query: str
    condensed_query: str
    answer: str
    citations: List[CitationResponse]
    retrieved_chunks: List[Dict[str, Any]]
    metadata: ExecutionMetadata

class IngestResponse(BaseModel):
    success: bool
    doc_id: str
    source: str
    pages_processed: int
    total_chunks_indexed: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    collection_name: str
    collection_count: int
    models: Dict[str, str]

# --- Query Condensation Helper (Pattern A) ---

async def condense_query(query: str, chat_history: List[ChatMessage]) -> str:
    """
    Evaluates raw query and preceding chat history. If history is present,
    invokes the fast model to rewrite the pronoun-dependent follow-up query
    into an explicit, search-optimized standalone question.
    """
    if not chat_history:
        return query

    history_text = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in chat_history])
    prompt = f"""Given the following conversation history and follow-up query, rewrite the follow-up query into a single standalone, search-optimized question.
The standalone question must resolve any ambiguous pronouns (such as "it", "this", "that", "they") based strictly on the chat history.
Do not answer the question; only output the rewritten standalone question.

Conversation History:
{history_text}

Follow-up Query: {query}

Standalone Question:"""

    try:
        fast_llm = get_fast_llm(temperature=0.0)
        response = await fast_llm.ainvoke([HumanMessage(content=prompt)])
        condensed = str(response.content).strip()
        print(f"[Query Condenser] Raw: '{query}' -> Condensed: '{condensed}'")
        return condensed
    except Exception as e:
        print(f"[Query Condenser Fallback] Using raw query due to: {e}")
        return query

# --- REST Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Returns service health, vector collection chunk count, and configured models."""
    db = get_database()
    count = await db.get_collection_count_async()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        collection_name=settings.collection_name,
        collection_count=count,
        models={
            "fast_llm": settings.fast_llm_model,
            "generation_llm": settings.generation_llm_model,
            "embedding": settings.embedding_model
        }
    )

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document_file(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    """
    Upload and index a PDF or Markdown document into the SOTA RAG database.
    Performs layout rendering, local image caching, and hierarchical indexing.
    """
    doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
    filename = file.filename or "uploaded_document"
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in [".pdf", ".md", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: .pdf, .md, .txt"
        )

    # Save to temporary file for parsing
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        stats = await ingest_file(
            file_path=temp_path,
            document_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return IngestResponse(
            success=True,
            doc_id=doc_id,
            source=filename,
            pages_processed=stats.get("pages_processed", 0),
            total_chunks_indexed=stats.get("total_chunks_indexed", 0),
            message=f"Successfully indexed document '{filename}' with {stats.get('total_chunks_indexed', 0)} chunks."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_agent(request: QueryRequest):
    """
    Stateless Query Endpoint.
    Applies Pattern A query condensation, executes the LangGraph corrective RAG pipeline,
    and returns context-grounded Markdown answers with structured inline citations.
    """
    start_time = time.perf_counter()

    # 1. Condense conversational query if history exists
    condensed_query = await condense_query(request.query, request.chat_history or [])

    # 2. Initialize graph state
    initial_state = {
        "raw_query": request.query,
        "query": condensed_query,
        "condensed_query": condensed_query,
        "retrieved_chunks": [],
        "route_decision": "retrieve",
        "retry_count": 0,
        "critique": None,
        "expanded_query": None,
        "is_relevant": None,
        "llm_inputs": [],
        "answer": None,
        "citations": [],
        "is_grounded": None,
        "groundedness_score": None,
        "metadata_filter": request.metadata_filter
    }

    try:
        # 3. Execute LangGraph workflow
        final_state = await graph.ainvoke(initial_state)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        answer = final_state.get("answer") or "Could not generate an answer."
        citations_data = final_state.get("citations", [])
        
        citations_response = [
            CitationResponse(
                id=c.get("id", idx + 1),
                source=c.get("source", "Document"),
                page=c.get("page"),
                doc_id=c.get("doc_id"),
                snippet=c.get("snippet", ""),
                image_url=c.get("image_url")
            )
            for idx, c in enumerate(citations_data)
        ]

        retrieved_chunks_out = [
            {
                "content": c.get("content", ""),
                "metadata": c.get("metadata", {}),
                "score": c.get("score")
            }
            for c in final_state.get("retrieved_chunks", [])
        ]

        metadata = ExecutionMetadata(
            retry_count=final_state.get("retry_count", 0),
            latency_ms=round(latency_ms, 2),
            is_relevant=final_state.get("is_relevant"),
            is_grounded=final_state.get("is_grounded"),
            groundedness_score=final_state.get("groundedness_score"),
            critique=final_state.get("critique")
        )

        return QueryResponse(
            success=True,
            raw_query=request.query,
            condensed_query=condensed_query,
            answer=answer,
            citations=citations_response,
            retrieved_chunks=retrieved_chunks_out,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent RAG workflow: {str(e)}"
        )

@app.post("/api/v1/query/stream")
async def query_rag_agent_stream(request: QueryRequest):
    """
    Server-Sent Events (SSE) Streaming Endpoint.
    Streams real-time LangGraph step transitions, judge critiques, and token chunks.
    """
    async def sse_event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()

        def format_sse(event_type: str, data: Dict[str, Any]) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        yield format_sse("start", {"raw_query": request.query, "timestamp": time.time()})

        # 1. Condensation
        condensed = await condense_query(request.query, request.chat_history or [])
        yield format_sse("query_condensed", {"condensed_query": condensed})

        initial_state = {
            "raw_query": request.query,
            "query": condensed,
            "condensed_query": condensed,
            "retrieved_chunks": [],
            "route_decision": "retrieve",
            "retry_count": 0,
            "critique": None,
            "expanded_query": None,
            "is_relevant": None,
            "llm_inputs": [],
            "answer": None,
            "citations": [],
            "is_grounded": None,
            "groundedness_score": None,
            "metadata_filter": request.metadata_filter
        }

        latest_state = initial_state

        try:
            # Stream node transitions
            async for output in graph.astream(initial_state):
                for node_name, node_state in output.items():
                    latest_state.update(node_state)
                    
                    if node_name == "retrieve":
                        chunks_summary = [
                            {"source": c.get("metadata", {}).get("source"), "page": c.get("metadata", {}).get("page")}
                            for c in node_state.get("retrieved_chunks", [])
                        ]
                        yield format_sse("retrieving", {
                            "node": "retrieve",
                            "chunks_found": len(chunks_summary),
                            "chunks": chunks_summary
                        })

                    elif node_name == "evaluate":
                        yield format_sse("evaluating", {
                            "node": "evaluate",
                            "is_relevant": node_state.get("is_relevant"),
                            "critique": node_state.get("critique"),
                            "expanded_query": node_state.get("expanded_query"),
                            "route_decision": node_state.get("route_decision"),
                            "retry_count": node_state.get("retry_count")
                        })

                    elif node_name == "assemble":
                        citations = [
                            {"id": c.get("id"), "source": c.get("source"), "page": c.get("page")}
                            for c in node_state.get("citations", [])
                        ]
                        yield format_sse("multimodal_assembly", {
                            "node": "assemble",
                            "citations": citations
                        })

                    elif node_name == "generate":
                        answer_text = node_state.get("answer", "")
                        # Stream the answer in simulated token chunks if generated in bulk
                        words = answer_text.split(" ")
                        for i in range(0, len(words), 4):
                            token_batch = " ".join(words[i:i+4]) + " "
                            yield format_sse("token", {"chunk": token_batch})
                            await asyncio.sleep(0.01)

                    elif node_name == "verify":
                        yield format_sse("verifying", {
                            "node": "verify",
                            "is_grounded": node_state.get("is_grounded"),
                            "groundedness_score": node_state.get("groundedness_score"),
                            "critique": node_state.get("critique")
                        })

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            yield format_sse("final_result", {
                "answer": latest_state.get("answer", ""),
                "citations": latest_state.get("citations", []),
                "metadata": {
                    "latency_ms": round(latency_ms, 2),
                    "retry_count": latest_state.get("retry_count", 0),
                    "is_grounded": latest_state.get("is_grounded"),
                    "groundedness_score": latest_state.get("groundedness_score")
                }
            })
            yield format_sse("done", {"status": "completed"})

        except Exception as err:
            yield format_sse("error", {"message": str(err)})

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=True)
