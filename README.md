# RAG-Ultra: SOTA Multi-Modal Agentic RAG Microservice (RaaS)

RAG-Ultra is a production-grade, minimal, and state-of-the-art (SOTA) **Retrieval-as-a-Service (RaaS) microservice** built in Python utilizing **LangGraph** (v1.2+) and **LangChain** (v1.3+). It is designed to operate as a completely stateless, high-performance REST API that parent conversational chatbots, agent teams, or background workflows can query over HTTP to delegate advanced, layout-aware document intelligence and multi-modal reasoning.

This microservice integrates cutting-edge 2026 engineering principles: **DeepSeek-OCR-2 API** for structural layout parsing, **Anthropic's Contextual Retrieval** (page-level metadata enrichment), **Single-Database Parent Payload Storage**, and a **LangGraph agentic correction & conditional multimodal generation loop** (GPT-5.5 / Gemini 3.5).

---

## 🚀 Key Architectural Innovations

### 1. Ingestion: Layout-Aware Parsing via DeepSeek-OCR-2
Traditional OCR parsers lose document structure (columns, tables, math formatting). RAG-Ultra utilizes a serverless API hosting **DeepSeek-OCR-2** to natively extract document layers into clean, structured Markdown — fully preserving LaTeX mathematical formulas ($\sum_{{i=1}}^n i = \frac{{n(n+1)}}{{2}}$), nested lists, and column hierarchies.

### 2. Semantic Enrichment: Contextual Retrieval
Small text chunks (e.g. 400 characters) are great for precise vector similarity search but lose overall document context. Using Anthropic's **Contextual Retrieval** pattern, a fast, low-cost model (`gpt-5.5-instant`) generates a 1-sentence page summary that is prepended as a global prefix to every child chunk before embedding, dramatically increasing matching relevance.

### 3. Storage Efficiency: Single-Database Parent Payloads
Traditional hierarchical chunking requires deploying, querying, and synchronizing a Vector Database (for children) and an external Key-Value Store (for parent documents). 
RAG-Ultra implements the **Single-Database Pattern**: child chunks are embedded in **Chroma**, but the complete parent page Markdown and secure page image URLs are written **directly inside the child's metadata payload**. This cuts database lookup network latency in half and eliminates DB synchronization bugs.

### 4. Orchestration: LangGraph Agentic Self-Correction
Rather than utilizing a static, linear retrieve-and-generate chain, the Real-Time query pipeline runs on a compiled **LangGraph State Graph**:
1. **Retrieve Node**: Conducts similarity search, with dynamic query expansion run on retries.
2. **LLM-as-a-Judge Node**: An evaluator (`gpt-5.5-instant`) inspects chunk relevance. If irrelevant, it triggers query correction and loops back to retry.
3. **Conditional Multimodal Node**: Analyzes payload metadata. If a parent page is text-only, it injects only the parent Markdown. If visual objects (charts, plots, blueprints) are marked, it downloads and base64-encodes the parent image, invoking **full visual reasoning** only when necessary to save VRAM and token overhead.
4. **Generator Node**: Generates the final, grounded answer using **`gpt-5.5`** or **`gemini-3.5-flash`**.

---

## 📊 System Topology

```
========================================================================================
1. INGESTION PIPELINE (Asynchronous / Batch)
========================================================================================
[PDF / Image Upload]
         |
         v
[PyMuPDF Page Splitter] --> Normalizes pages to clean JPEG frames (150 DPI)
         |
         v
[DeepSeek-OCR Tool]     --> Serverless DeepSeek-OCR-2 API parses page into Markdown
         |                  └--> Detects if page has visual elements (charts, diagrams)
         |
         v
[Contextualizer Node]   --> gpt-5.5-instant generates 1-sentence page summary
         |
         v
[Hierarchical Splitter] --> Splits page into Parents (2000 chars) and Children (400 chars)
         |                  └--> Prepends contextual prefix to child chunks
         |
         v
[Vector Database]       --> Embeds child chunks. Stores parent markdown AND raw page image
                            payload in Chroma (Single-Database Pattern).

========================================================================================
2. RETRIEVAL & REASONING PIPELINE (LangGraph Real-Time Execution)
========================================================================================
                  [ User Prompt Input ]
                            |
                            v
                  [ Retrieve Node ] <----------------------------------+
                  | 1. Vector similarity search on child chunks.      |
                  | 2. Dynamic query expansion on retrieval retries.  |
                  +----------+-----------------------------------------+
                             |
                             v
                  [ Relevance Evaluator Node ] --(Irrelevant / Retry)--+
                  | (LLM-as-a-Judge inspects top chunks vs. query)
                  +----------+------------------------------------------
                             | (Relevant / Sufficient Context)
                             v
                  [ Conditional Multimodal Node ]
                  | Check if retrieved chunks contain visual references.
                  | If yes: Download/base64 encode parent page image.
                  | If no: Bypass image download (saves latency/cost).
                  +----------+------------------------------------------
                             |
                             v
                  [ Generation Node ]
                  | Generates final context-grounded answer via GPT-5.5 / Gemini 3.5.
                  +------------------------------------------
```

---

## 📂 Project Directory Structure

```text
rag-ultra/
│
├── my_agent/                 # Compiled LangGraph Workflow
│   ├── utils/                # Graph Helpers & State Schemas
│   │   ├── __init__.py
│   │   ├── state.py          # AgentState Definition
│   │   ├── tools.py          # deepseek_ocr_parse & vector_search_db tools
│   │   └── nodes.py          # retrieve, evaluate, assemble, and generate nodes
│   ├── __init__.py
│   └── agent.py              # Constructs and compiles StateGraph
│
├── core/                     # Ingestion Logic Core
│   ├── __init__.py
│   ├── database.py           # SotaRagDatabase (Chroma parent-payload wrapper)
│   └── contextualizer.py     # Contextual Retrieval summarizer
│
├── .env                      # Environment Variables Template
├── langgraph.json            # LangGraph CLI config file
├── pyproject.toml            # Dependencies specification
├── app.py                    # REST API FastAPI gateway server
└── ingest_cli.py             # Document Ingestion CLI script
```

---

## 🛠️ Setup & Installation

The project uses the fast, modern **`uv`** package manager.

### 1. Clone the repository and sync dependencies:
```bash
# Installs Python virtual environment and all requirements
uv sync
```

### 2. Configure Environment Variables:
Copy the template `.env` file and insert your API credentials:
```bash
# Rename or edit .env
# Set your OpenAI key (required for embeddings and generation)
OPENAI_API_KEY=your_openai_api_key_here

# Set your Novita AI key (required for DeepSeek-OCR API)
NOVITA_API_KEY=your_novita_api_key_here
NOVITA_API_URL=https://api.novita.ai/v1/chat/completions
```

---

## 🚀 Quickstart Guide

### Step 1: Ingest a Document
Normalize and parse an operational handbook or PDF. This splits the pages, invokes DeepSeek-OCR, generates page-level summaries, chunks, and embeds them into Chroma:
```bash
uv run python ingest_cli.py --pdf path/to/your/document.pdf --id manual_001
```

### Step 2: Launch the FastAPI Backend

Start the REST API gateway:

```bash
uv run python app.py
```

The query endpoint is designed to be completely **stateless**. Since RAG-Ultra utilizes real-time query condensation, there is **no need to supply a `thread_id`** or manage state checkpointers in the database. Instead, the parent chatbot simply passes the preceding conversational log directly in the `chat_history` payload, and the gateway automatically condenses follow-up queries before searching:

```http
POST /api/v1/query
Content-Type: application/json
```

```json
{
  "query": "Is it valid in wet conditions?",
  "chat_history": [
    {
      "role": "user",
      "content": "What is the safety checklist for operating Valve A?"
    },
    {
      "role": "assistant",
      "content": "The safety checklist for Valve A requires dry conditions, insulation tools, and gloves."
    }
  ]
}
```

---

## 📝 Observability & Tracing

Observability is natively supported using **LangSmith**.
To enable granular visual tracing, tool execution metrics, and node transition graphs, simply sync these variables in your `.env`:
```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-ultra-agent
```
Once enabled, every query run will be instantly traceable inside your LangSmith dashboard, allowing you to review LLM-as-a-Judge outputs, token counts, and API response speeds.
