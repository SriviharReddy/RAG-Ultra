# Ingestion Engine & Layout-Aware OCR

RAG-Ultra includes a dedicated ingestion engine capable of parsing complex technical PDFs, Markdown files, and text documents while preserving visual diagrams, Markdown tables, and mathematical formulas.

---

## 1. Pipeline Overview

```text
[ Document File (PDF / MD) ]
             |
             v
+----------------------------+
| 1. Page Normalizer (PyMuPDF)| --> Renders 150 DPI JPEG images
+----------------------------+     Saves to ./db_storage/images/{doc_id}/page_{N}.jpg
             |
             v
+----------------------------+
| 2. Vision OCR & Extraction | --> Novita AI (Qwen-2.5-VL) / OpenAI Vision (gpt-4o-mini)
+----------------------------+     Zero-crash fallback to PyMuPDF native text extractor
             |
             v
+----------------------------+
| 3. Contextual Enrichment   | --> Generates Anthropic-style 1-sentence page prefix
+----------------------------+     [Context: Context from 'Doc Title' (Page N): Section Summary]
             |
             v
+----------------------------+
| 4. Layout-Aware Splitter   | --> RecursiveCharacterTextSplitter
+----------------------------+     Preserves headers (\n## ), Markdown tables (| col |), and math
             |
             v
+----------------------------+
| 5. Chroma Vector Indexer   | --> Stores child chunks with embedded parent text & image URIs
+----------------------------+
```

---

## 2. Supported Vision OCR Providers

RAG-Ultra automatically selects the most capable OCR engine available based on your environment configuration:

### Provider Priority Order:
1. **Novita AI API** (`NOVITA_API_KEY`):
   - Uses hosted open-weights Vision Language Models such as `qwen/qwen-2.5-vl-72b-instruct` or `deepseek-vl`.
   - Endpoint: `https://api.novita.ai/v1`
2. **OpenAI Vision API** (`OPENAI_API_KEY`):
   - Uses `gpt-4o-mini` (or `gpt-4o`) with high-detail image parsing.
   - Requires no additional credentials if OpenAI is already configured for the agent.
3. **Custom OpenAI-Compatible VLM** (`OCR_BASE_URL` + `OCR_API_KEY`):
   - Compatible with OpenRouter, Together AI, Ollama, or vLLM deployments.
4. **PyMuPDF Native Layout Extractor** (Offline Fallback):
   - If no API keys are provided or network errors occur, PyMuPDF extracts text, detected tables, and structural blocks directly with zero network calls.

---

## 3. CLI Ingestion Tool

Run `ingest_cli.py` to index documents directly from the terminal:

```bash
# Ingest a PDF document
uv run python ingest_cli.py --file docs/turbine_manual.pdf --id manual_001

# Custom chunk size and overlap
uv run python ingest_cli.py \
  --file docs/handbook.pdf \
  --id handbook_2026 \
  --chunk-size 600 \
  --chunk-overlap 100

# Ingest a Markdown document
uv run python ingest_cli.py --file README.md --id readme_doc
```

### CLI Arguments:
- `--file` / `--pdf`: Path to the input file (`.pdf`, `.md`, `.txt`).
- `--id`: Document unique identifier string (default: `doc_001`).
- `--chunk-size`: Character limit per child chunk (default: `800`).
- `--chunk-overlap`: Overlap between consecutive chunks (default: `100`).

---

## 4. Programmatic Python Usage

```python
import asyncio
from ingest_cli import ingest_file

async def index_manual():
    stats = await ingest_file(
        file_path="./data/alpha9_manual.pdf",
        document_id="alpha9_v1",
        chunk_size=700,
        chunk_overlap=100
    )
    print(f"Indexed {stats['total_chunks_indexed']} chunks across {stats['pages_processed']} pages.")

if __name__ == "__main__":
    asyncio.run(index_manual())
```
