# RAG-Ultra: SOTA Multi-Modal Agentic RAG Microservice (RaaS)

RAG-Ultra is a production-grade, state-of-the-art **Retrieval-as-a-Service (RaaS) microservice** built in Python utilizing **LangGraph** (v1.2+), **FastAPI**, and **LangChain** (v1.3+). It operates as a stateless, high-performance REST and Server-Sent Events (SSE) API that client applications, parent agent teams, and conversational bots query over HTTP to delegate advanced, layout-aware document intelligence and multimodal reasoning.

The microservice integrates cutting-edge Agentic AI engineering principles:
1. **Layout-Aware Ingestion & Local Image Caching**: High-fidelity page extraction with Vision OCR (Novita AI, OpenAI Vision, or custom VLM endpoint) and zero-crash PyMuPDF native fallbacks.
2. **Anthropic Contextual Retrieval**: Prepends page-level metadata summaries to child chunks for high-recall vector search.
3. **Single-Database Parent Payloads**: Stores complete parent Markdown and page image paths directly within Chroma metadata, cutting latency and eliminating dual-store synchronization.
4. **Structured Corrective RAG (CRAG)**: Pydantic-powered LLM-as-a-Judge relevance grading and query expansion.
5. **Reciprocal Rank Fusion (RRF)**: Merges retried search iterations with prior hits to prevent context loss.
6. **Conditional Multimodal Context Assembly**: Asynchronously loads and base64-encodes page diagrams only when visuals exist.
7. **Groundedness Verification & Self-Correction**: Checks generated answers against retrieved context to prevent hallucinations.
8. **Real-time Streaming (SSE)**: Streams graph node lifecycle transitions and token chunks for live agent telemetry.

---

## 🚀 Key Architectural Patterns

### 1. Ingestion Engine & Layout-Aware Splitter
- Converts PDF documents into normalized page frames rendered at 150 DPI and cached locally in `./db_storage/images/{doc_id}/page_{page_num}.jpg`.
- Extracts structured Markdown via **Vision OCR** (Novita AI `qwen-2.5-vl`, OpenAI Vision `gpt-4o-mini`, or custom VLM) with graceful fallback to **PyMuPDF native page extractors** for offline execution.
- Employs **Recursive Markdown Chunking** (`RecursiveCharacterTextSplitter`), preserving Markdown tables, headers, and code fences.

### 2. Semantic Contextual Retrieval
Small text chunks (e.g. 500-800 characters) are optimal for dense vector matching but lose document context. RAG-Ultra generates a concise 1-sentence contextual overlay for each page, prepending `[Context: <summary>]\n<chunk>` prior to embedding.

### 3. Single-Database Parent Payloads
Traditional hierarchical chunking requires separate vector and key-value stores. RAG-Ultra embeds child chunks in **Chroma** while storing the full parent page Markdown, image paths, visual flags, and provenance metadata directly inside the child record's payload.

### 4. LangGraph Corrective Agent Loop
The inference workflow executes as a compiled LangGraph state machine:
- **`retrieve`**: Asynchronous vector similarity search supporting metadata filters and query expansion.
- **`evaluate`**: Structured `GradeEvaluation` Pydantic LLM-as-a-Judge. Loops back to `retrieve` if context is insufficient.
- **`assemble`**: Concurrently loads local/remote visual page images, deduplicates context blocks, and formats structured inline citations (`[^1]`).
- **`generate`**: Synthesizes a grounded response via multimodal LLM.
- **`verify`**: Evaluates `GroundednessEvaluation`. Initiates corrective retrieval if hallucinations are detected.

---

## 📊 System Topology

```text
========================================================================================
1. INGESTION PIPELINE (Batch / Multipart Upload)
========================================================================================
[PDF / Markdown Document]
         |
         v
[PyMuPDF Page Splitter]    --> Normalizes pages to JPEG frames (150 DPI) in ./db_storage/images/
         |
         v
[OCR Engine / Fallback]    --> Vision OCR API (Novita AI / OpenAI Vision / PyMuPDF native extractor)
         |                     └--> Detects tables, charts, diagrams (has_visuals: True/False)
         v
[Contextualizer Node]      --> Generates 1-sentence page contextual overlay
         |
         v
[Recursive MD Splitter]    --> Markdown-aware splitting preserving tables & section headers
         |
         v
[Chroma Vector Database]   --> Ingests child chunks with parent markdown & image path payloads

========================================================================================
2. AGENTIC RETRIEVAL & REASONING PIPELINE (LangGraph Execution)
========================================================================================
             [ User Query / Chat History ]
                         |
                         v
             [ Pattern A Query Condenser ]
                         |
                         v
+-----------> [ Retrieve Node ] <----------------------------------------+
|             | 1. Async similarity search in Chroma.                   |
|             | 2. Reciprocal Rank Fusion (RRF) on retry merges.        |
|             +----------+----------------------------------------------+
|                        |
|                        v
|             [ Relevance Evaluator Node ] (LLM-as-a-Judge)
|             | Structured Pydantic GradeEvaluation (is_relevant, critique)
|             +----------+----------------------------------------------+
|                        |
|        (Relevant / Max Retries)               (Insufficient Context)
|                        |                                 |
|                        v                                 |
|             [ Multimodal Assembly Node ]                 |
|             | Deduplicates parent context blocks.        |
|             | Concurrently loads & encodes visual images.|
|             | Generates structured inline citations [^N].|
|             +----------+---------------------------------+
|                        |
|                        v
|             [ Generation Node ]
|             | Synthesizes answer using Multimodal LLM.
|             +----------+---------------------------------+
|                        |
|                        v
+------------ [ Groundedness Verifier Node ] (Self-Correction)
(Unhallucinated)   | Pydantic GroundednessEvaluation (is_grounded, score)
   Loopback       +----------+---------------------------------+
                             | (Verified Grounded / Max Retries)
                             v
                         [ [END] ] -> Final Markdown Response with Footnotes
```

---

## 📂 Project Directory Structure

```text
rag-ultra/
│
├── core/
│   ├── __init__.py
│   ├── config.py             # Centralized Pydantic settings & LLM provider factories
│   ├── database.py           # SotaRagDatabase Chroma async manager with parent payloads
│   └── contextualizer.py     # Contextual Retrieval summarizer with offline fallback
│
├── my_agent/                 # Compiled LangGraph Workflow
│   ├── __init__.py
│   ├── agent.py              # Compiled StateGraph with CRAG & Groundedness edges
│   └── utils/
│       ├── __init__.py
│       ├── state.py          # Typed AgentState, DocumentChunk, Citation schemas
│       ├── nodes.py          # retrieve, evaluate, assemble, generate, verify nodes
│       └── tools.py          # Vector search and DeepSeek OCR tool definitions
│
├── app.py                    # FastAPI Gateway (REST, SSE streaming, file ingestion)
├── ingest_cli.py             # CLI Ingestion tool (PDF/MD layout chunking)
├── demo.py                   # Self-contained showcase demo & verification suite
├── pyproject.toml            # Dependencies and virtual environment spec
└── README.md
```

---

## 🛠️ Setup & Installation

The project uses the fast **`uv`** package manager.

### 1. Install dependencies:
```bash
# Automatically creates .venv and installs all dependencies
uv sync
```

### 2. Configure Environment (`.env`):
Create or edit your `.env` file:
```ini
# OpenAI Configuration (Embeddings, Fast Judge, and Generation LLM)
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1

# Optional Vision OCR Provider (Novita AI / Custom VLM)
NOVITA_API_KEY=your_novita_api_key_here
NOVITA_BASE_URL=https://api.novita.ai/v1
NOVITA_MODEL=qwen/qwen-2.5-vl-72b-instruct
# Note: Official api.deepseek.com only provides text endpoints; open-weights VLMs (DeepSeek-VL, Qwen-VL)
# are hosted on serverless providers like Novita AI, OpenRouter, or handled directly via OpenAI Vision.
# Model Names
FAST_LLM_MODEL=gpt-4o-mini
GENERATION_LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Storage Directories
PERSIST_DIR=./db_storage/chroma
IMAGE_STORAGE_DIR=./db_storage/images
COLLECTION_NAME=sota_rag_collection
```

---

## 🚀 Running the End-to-End Showcase Demo

To execute the complete end-to-end verification suite with zero external infrastructure:
```bash
uv run python demo.py
```
This standalone script:
1. Generates a sample multi-page industrial manual (`turbine_alpha9_manual.pdf`) with technical diagrams and calibration tables.
2. Ingests the PDF, creates local image caches, generates contextual prefixes, and indexes parent-child payloads in Chroma.
3. Tests **Pattern A Conversational Query Condensation** ("What about in wet conditions?").
4. Executes the **LangGraph CRAG State Machine** with relevance grading, multimodal visual assembly, and inline citations.
5. Tests the **FastAPI Gateway**:
   - `GET /api/v1/health`
   - `POST /api/v1/ingest` (multipart file upload)
   - `POST /api/v1/query` (stateless query)
   - `POST /api/v1/query/stream` (real-time Server-Sent Events stream)

---

## 📡 REST API Gateway Reference

Start the production server:
```bash
uv run python app.py
```

### 1. Health Check
```http
GET /api/v1/health
```
**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "collection_name": "sota_rag_collection",
  "collection_count": 12,
  "models": {
    "fast_llm": "gpt-4o-mini",
    "generation_llm": "gpt-4o",
    "embedding": "text-embedding-3-small"
  }
}
```

### 2. Ingest Document (Multipart Upload)
```http
POST /api/v1/ingest
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: PDF or Markdown file binary.
- `document_id`: (Optional) Unique document string identifier.
- `chunk_size`: (Optional) Integer chunk character length (default: 800).
- `chunk_overlap`: (Optional) Integer chunk overlap (default: 100).

**Response:**
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

### 3. Stateless Query
```http
POST /api/v1/query
Content-Type: application/json
```
**Payload:**
```json
{
  "query": "What about in wet conditions?",
  "chat_history": [
    {
      "role": "user",
      "content": "What is the maximum pressure limit for Turbine Alpha-9?"
    },
    {
      "role": "assistant",
      "content": "The maximum nominal pressure is 450 PSI with emergency vent at 520 PSI."
    }
  ],
  "metadata_filter": {
    "doc_id": "manual_turbine_a9"
  }
}
```
**Response:**
```json
{
  "success": true,
  "raw_query": "What about in wet conditions?",
  "condensed_query": "What are the operating protocols and constraints for Turbine Alpha-9 in wet conditions?",
  "answer": "Operating in wet environments requires adherence to Protocol W-7 [^1]. Key requirements include:\n- Verifying IP67 waterproof enclosure seals [^1]\n- Derating continuous output by 15% when humidity exceeds 90% [^1]\n- Enabling automatic manifold heaters below 4°C [^1].",
  "citations": [
    {
      "id": 1,
      "source": "turbine_manual.pdf",
      "page": 2,
      "doc_id": "manual_turbine_a9",
      "snippet": "Turbine Alpha-9 Operating Manual Section 2: Wet Conditions and Flood Protocol...",
      "image_url": "/static/images/manual_turbine_a9/page_2.jpg"
    }
  ],
  "retrieved_chunks": [...],
  "metadata": {
    "retry_count": 0,
    "latency_ms": 842.15,
    "is_relevant": true,
    "is_grounded": true,
    "groundedness_score": 1.0,
    "critique": "Context fully supports all claims."
  }
}
```

### 4. Real-time SSE Streaming
```http
POST /api/v1/query/stream
Content-Type: application/json
```
Streams real-time Server-Sent Events (`text/event-stream`):
- `event: query_condensed` -> Emits standalone rewritten question.
- `event: retrieving` -> Emits candidate chunk count and source metadata.
- `event: evaluating` -> Emits LLM-as-a-Judge relevance verdict and critique.
- `event: multimodal_assembly` -> Emits loaded citations and visual image count.
- `event: token` -> Streams generated token increments.
- `event: verifying` -> Emits groundedness score and verification verdict.
- `event: final_result` -> Emits complete response object with citations and latency.
- `event: done` -> Final completion marker.

---

## 🛠️ CLI Ingestion Tool

You can ingest files directly via the CLI:
```bash
# Ingest PDF
uv run python ingest_cli.py --file docs/handbook.pdf --id handbook_v1 --chunk-size 600 --chunk-overlap 100

# Ingest Markdown
uv run python ingest_cli.py --file docs/architecture.md --id arch_doc
```

---

## 📝 Observability & Tracing

Granular observability is supported via **LangSmith**. Set the following environment variables:
```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-ultra-agent
```
Once enabled, every LangGraph node transition, LLM evaluation prompt, and retrieval score is visible in the LangSmith dashboard.
