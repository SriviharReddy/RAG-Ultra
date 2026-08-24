# RAG-Ultra: SOTA Multimodal Agentic RAG Microservice

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RAG-Ultra** is a production-grade **Retrieval-as-a-Service (RaaS) microservice** built with **LangGraph**, **FastAPI**, and **LangChain**. It exposes a stateless REST and Server-Sent Events (SSE) API designed to handle layout-aware document parsing, multimodal visual reasoning, and self-correcting agent loops.

---

## 🌟 Key Highlights

- **Layout-Aware Ingestion**: Parses PDFs and Markdown into clean, structured text preserving tables (`| col |`), LaTeX equations (`$...$`), and diagrams.
- **Anthropic Contextual Retrieval**: Prepends 1-sentence page-level context overlays to child chunks to boost semantic recall.
- **Single-Database Parent Payloads**: Stores complete parent page text and image URIs directly inside Chroma metadata, eliminating dual-store synchronization.
- **Corrective RAG (CRAG) with Judge Fast-Path**: Structured Pydantic LLM-as-a-Judge grading with an automatic fast-path for high-confidence matches ($\ge 0.82$).
- **Reciprocal Rank Fusion (RRF)**: Merges retried search iterations with prior hits to ensure no context is discarded.
- **Conditional Multimodal Assembly**: Loads and base64-encodes page diagrams only when visual graphics are present.
- **Groundedness Self-Correction**: Verifies generated answers against retrieved context to prevent hallucinations.
- **Real-Time SSE Streaming**: Live event stream (`text/event-stream`) for graph state transitions, judge evaluations, and token chunks.

---

## 📊 System Flow

```text
[ Document / PDF ] ──> [ Vision OCR & Splitter ] ──> [ Chroma (Parent Payloads) ]
                                                              │
[ Query + History ] ──> [ Query Condenser ] ──> [ Retrieve Node ] <───┐ (Retry Loop)
                                                      │               │
                                                      v               │
                                            [ LLM-as-a-Judge ] ───────┤
                                                      │ (Relevant)    │
                                                      v               │
                                            [ Multimodal Assembly ]   │
                                                      │               │
                                                      v               │
                                            [ Answer Generation ]     │
                                                      │               │
                                                      v               │
                                            [ Groundedness Check ] ───┘ (Self-Correction)
                                                      │
                                                      v
                                        [ Answer + Inline Citations [^1] ]
```

---

## 🚀 Quickstart

### 1. Installation
Install dependencies using `uv`:
```bash
uv sync
```

### 2. Configure Environment
Create your `.env` file (see [Configuration Guide](docs/configuration.md) for all options):
```ini
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Showcase Demo
Execute the self-contained verification suite (creates a test document, runs ingestion, query condensation, CRAG graph execution, and all API endpoints):
```bash
uv run python demo.py
```

### 4. Start the API Gateway
```bash
uv run python app.py
```

---

## 📡 API Quick Glance

### Query with Follow-up History
```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about in wet conditions?",
    "chat_history": [
      {
        "role": "user",
        "content": "What is the maximum pressure for Turbine Alpha-9?"
      },
      {
        "role": "assistant",
        "content": "The maximum nominal pressure is 450 PSI."
      }
    ]
  }'
```

### Real-Time SSE Stream
```bash
curl -N -X POST http://localhost:8080/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain Protocol W-7 safety checklist."}'
```

---

## 📚 Documentation Index

Detailed guides are available in the [`docs/`](docs/) directory:

- 🏗️ **[System Architecture & Design](docs/architecture.md)** — Deep dive into the LangGraph state machine, Single-Database Parent Payloads, CRAG loop, and RRF merging.
- 📡 **[API Gateway & Reference](docs/api-reference.md)** — Complete specification of all REST endpoints, SSE event streams, request/response schemas, and cURL examples.
- 📄 **[Ingestion Engine & Layout-Aware OCR](docs/ingestion-and-ocr.md)** — PDF page normalization, local image caching, multi-provider Vision OCR, and CLI ingestion commands.
- ⚙️ **[Configuration & Environment Guide](docs/configuration.md)** — Environment variables, model provider settings, storage paths, and LangSmith tracing.

---

## 📂 Project Structure

```text
rag-ultra/
│
├── docs/                     # Detailed modular documentation
│   ├── architecture.md       # State graph & architectural patterns
│   ├── api-reference.md      # REST & SSE endpoint specification
│   ├── ingestion-and-ocr.md  # Layout-aware chunking & Vision OCR
│   └── configuration.md      # Settings & environment parameters
│
├── core/                     # Core Backend Components
│   ├── config.py             # Pydantic settings & LLM factories
│   ├── database.py           # Thread-safe async Chroma parent-payload wrapper
│   └── contextualizer.py     # Contextual Retrieval summarizer
│
├── my_agent/                 # Compiled LangGraph Workflow
│   ├── agent.py              # StateGraph with CRAG & verification edges
│   └── utils/                # Nodes, state schemas, and tools
│
├── app.py                    # FastAPI Gateway (REST & SSE streaming)
├── ingest_cli.py             # CLI Ingestion tool (PDF & Markdown)
├── demo.py                   # Self-contained showcase demo & verification suite
└── README.md
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
