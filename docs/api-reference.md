# API Reference & Gateway Specification

The RAG-Ultra API Gateway is built on **FastAPI** and provides stateless REST endpoints and real-time Server-Sent Events (SSE) streaming for agent workflows.

---

## Base URL
```text
http://localhost:8080
```

---

## 1. Health & Status Check

### `GET /api/v1/health`
Checks vector database readiness, active collection chunk count, and model provider configurations.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "collection_name": "sota_rag_collection",
  "collection_count": 42,
  "models": {
    "fast_llm": "gpt-4o-mini",
    "generation_llm": "gpt-4o",
    "embedding": "text-embedding-3-small"
  }
}
```

#### Example cURL:
```bash
curl -X GET http://localhost:8080/api/v1/health
```

---

## 2. Document Ingestion

### `POST /api/v1/ingest`
Uploads and indexes a PDF or Markdown document. Performs page normalization, image caching, layout extraction, contextual prefixing, and Chroma indexing.

#### Request (`multipart/form-data`)
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | Binary File | **Yes** | PDF (`.pdf`), Markdown (`.md`), or Text (`.txt`) file. |
| `document_id` | String | No | Custom document ID (auto-generated if omitted). |
| `chunk_size` | Integer | No | Character length per chunk (default: `800`). |
| `chunk_overlap` | Integer | No | Overlap between chunks (default: `100`). |

#### Response (`200 OK`)
```json
{
  "success": true,
  "doc_id": "manual_turbine_a9",
  "source": "turbine_manual.pdf",
  "pages_processed": 2,
  "total_chunks_indexed": 6,
  "message": "Successfully indexed document 'turbine_manual.pdf' with 6 chunks."
}
```

#### Example cURL:
```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@/path/to/handbook.pdf" \
  -F "document_id=handbook_001" \
  -F "chunk_size=600" \
  -F "chunk_overlap=80"
```

---

## 3. Stateless Query

### `POST /api/v1/query`
Executes the full agentic Corrective RAG pipeline. If `chat_history` is supplied, Pattern A conversational query condensation is automatically applied.

#### Request Body (`application/json`)
```json
{
  "query": "What about in wet conditions?",
  "chat_history": [
    {
      "role": "user",
      "content": "What is the maximum operating pressure for Turbine Alpha-9?"
    },
    {
      "role": "assistant",
      "content": "The maximum nominal pressure is 450 PSI, with an emergency vent trigger at 520 PSI."
    }
  ],
  "metadata_filter": {
    "doc_id": "manual_turbine_a9"
  }
}
```

#### Response (`200 OK`)
```json
{
  "success": true,
  "raw_query": "What about in wet conditions?",
  "condensed_query": "What are the operating protocols and constraints for Turbine Alpha-9 in wet conditions?",
  "answer": "Operating in wet environments requires adherence to Protocol W-7 [^1]. Key requirements include:\n- Verifying IP67 waterproof enclosure seals on all electrical junctions [^1]\n- Derating maximum continuous output by 15% when humidity exceeds 90% [^1]\n- Enabling automatic manifold heaters below 4°C [^1].",
  "citations": [
    {
      "id": 1,
      "source": "turbine_alpha9_manual.pdf",
      "page": 2,
      "doc_id": "manual_turbine_a9",
      "snippet": "Turbine Alpha-9 Operating Manual Section 2: Wet Conditions and Flood Protocol...",
      "image_url": "/static/images/manual_turbine_a9/page_2.jpg"
    }
  ],
  "retrieved_chunks": [
    {
      "content": "...",
      "metadata": {
        "source": "turbine_alpha9_manual.pdf",
        "page": 2,
        "doc_id": "manual_turbine_a9",
        "has_visuals": true
      },
      "score": 1.48
    }
  ],
  "metadata": {
    "retry_count": 0,
    "latency_ms": 654.73,
    "is_relevant": true,
    "is_grounded": true,
    "groundedness_score": 1.0,
    "critique": "High confidence similarity match (Score: 1.479)."
  }
}
```

#### Example cURL:
```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the inspection frequency for Bearings-A?",
    "chat_history": []
  }'
```

---

## 4. Real-Time Streaming (SSE)

### `POST /api/v1/query/stream`
Streams real-time Server-Sent Events (`text/event-stream`) representing graph execution states and token increments.

#### Request Body (`application/json`)
```json
{
  "query": "What is Protocol W-7 for wet conditions?",
  "chat_history": []
}
```

#### Event Stream Lifecycle:
1. **`event: start`**: Workflow invocation timestamp.
2. **`event: query_condensed`**: Standalone rewritten query.
3. **`event: retrieving`**: Number and sources of candidate chunks retrieved.
4. **`event: evaluating`**: LLM-as-a-Judge relevance verdict, critique, and route decision.
5. **`event: multimodal_assembly`**: Loaded inline citations and visual diagram count.
6. **`event: token`**: Streamed response token chunks.
7. **`event: verifying`**: Groundedness verification score and critique.
8. **`event: final_result`**: Complete JSON payload with answer, citations, and execution latency.
9. **`event: done`**: Stream completion signal.

#### Example Stream Output:
```text
event: start
data: {"raw_query": "What is Protocol W-7?", "timestamp": 1724500000.12}

event: query_condensed
data: {"condensed_query": "What is Protocol W-7 for wet conditions?"}

event: retrieving
data: {"node": "retrieve", "chunks_found": 3, "chunks": [{"source": "manual.pdf", "page": 2}]}

event: evaluating
data: {"node": "evaluate", "is_relevant": true, "critique": "High confidence similarity match."}

event: multimodal_assembly
data: {"node": "assemble", "citations": [{"id": 1, "source": "manual.pdf", "page": 2}]}

event: token
data: {"chunk": "Based on the retrieved "}

event: token
data: {"chunk": "documentation [^1]: "}

event: verifying
data: {"node": "verify", "is_grounded": true, "groundedness_score": 1.0}

event: final_result
data: {"answer": "...", "citations": [...], "metadata": {...}}

event: done
data: {"status": "completed"}
```

#### Example cURL:
```bash
curl -N -X POST http://localhost:8080/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Protocol W-7?"}'
```
