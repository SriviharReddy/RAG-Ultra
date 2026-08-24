# Configuration & Environment Guide

RAG-Ultra uses **Pydantic Settings** (`core/config.py`) to manage application settings with fallback defaults and environment overrides.

---

## 1. Environment Variables Template (`.env`)

```ini
# ==============================================================================
# Model Providers & API Credentials
# ==============================================================================
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1

# Optional Vision OCR Provider (Novita AI or custom VLM)
NOVITA_API_KEY=your_novita_api_key_here
NOVITA_BASE_URL=https://api.novita.ai/v1
NOVITA_MODEL=qwen/qwen-2.5-vl-72b-instruct

# Custom OpenAI-compatible OCR Endpoint (Optional)
# OCR_API_KEY=your_custom_api_key
# OCR_BASE_URL=https://openrouter.ai/api/v1
# OCR_MODEL=qwen/qwen-2.5-vl-72b-instruct

# ==============================================================================
# Model Selection
# ==============================================================================
FAST_LLM_MODEL=gpt-4o-mini
GENERATION_LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# ==============================================================================
# Vector Store & Storage Directories
# ==============================================================================
PERSIST_DIR=./db_storage/chroma
IMAGE_STORAGE_DIR=./db_storage/images
COLLECTION_NAME=sota_rag_collection

# ==============================================================================
# Execution Parameters
# ==============================================================================
MAX_RETRIES=3
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=3

# ==============================================================================
# Server Gateway
# ==============================================================================
HOST=0.0.0.0
PORT=8080

# ==============================================================================
# Observability & Tracing (LangSmith)
# ==============================================================================
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-ultra-agent
```

---

## 2. Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | String | `None` | OpenAI API key for generation, fast judge, and embeddings. |
| `FAST_LLM_MODEL` | String | `gpt-4o-mini` | Low-latency model for query condensation, grading, and verification. |
| `GENERATION_LLM_MODEL` | String | `gpt-4o` | Multimodal model for final answer synthesis. |
| `EMBEDDING_MODEL` | String | `text-embedding-3-small` | OpenAI embedding model for vector search. |
| `NOVITA_API_KEY` | String | `None` | Optional API key for hosted open-weights Vision OCR models. |
| `PERSIST_DIR` | String | `./db_storage/chroma` | Local directory for Chroma vector database persistence. |
| `IMAGE_STORAGE_DIR` | String | `./db_storage/images` | Local directory where normalized 150 DPI page frames are saved. |
| `COLLECTION_NAME` | String | `sota_rag_collection` | Chroma collection name. |
| `MAX_RETRIES` | Integer | `3` | Maximum Corrective RAG query reformulation retries. |
| `CHUNK_SIZE` | Integer | `800` | Target character size for child text chunks. |
| `CHUNK_OVERLAP` | Integer | `100` | Overlap character size between adjacent child chunks. |
| `TOP_K` | Integer | `3` | Number of chunks to retrieve during vector search. |
| `PORT` | Integer | `8080` | HTTP port for FastAPI server. |

---

## 3. Observability & LangSmith Tracing

To monitor graph transitions, token latency, and judge critiques in production:
1. Enable `LANGSMITH_TRACING=true` in your `.env`.
2. Add your `LANGSMITH_API_KEY`.
3. Set `LANGSMITH_PROJECT=rag-ultra-agent`.

Every call to `POST /api/v1/query` or `POST /api/v1/query/stream` will automatically publish granular trace trees to your LangSmith dashboard.
