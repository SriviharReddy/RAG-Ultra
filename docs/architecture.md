# System Architecture & Design

RAG-Ultra is an agentic **Retrieval-as-a-Service (RaaS)** microservice designed for production-grade document intelligence and multimodal reasoning.

---

## 1. High-Level System Topology

```text
========================================================================================
1. INGESTION ENGINE (Layout-Aware Pipeline)
========================================================================================
[PDF / Markdown Upload]
         |
         v
[PyMuPDF Page Normalizer] --> Render 150 DPI JPEG frames to ./db_storage/images/{doc_id}/
         |
         v
[Vision OCR / Extractor]  --> Novita AI (Qwen-2.5-VL) / OpenAI Vision / PyMuPDF fallback
         |                     └--> Detects tables, charts, figures (has_visuals: True/False)
         v
[Contextual Retrieval]    --> Anthropic-style 1-sentence page context prefix with XML tags
         |
         v
[Recursive MD Splitter]   --> Preserves Markdown tables (| col |), headers, code fences
         |
         v
[Chroma Vector Store]     --> Embeds child chunks; writes parent Markdown and image path
                              directly into child metadata (Single-Database Pattern).

========================================================================================
2. INFERENCE & AGENTIC REASONING (LangGraph State Machine)
========================================================================================
                 [ User Query + Chat History ]
                               |
                               v
                 [ Pattern A Query Condenser ]
                               |
                               v
+-------------> [ Retrieve Node ] <------------------------------------+
|               | 1. Async Chroma similarity search (top-k = 3).       |
|               | 2. Reciprocal Rank Fusion (RRF) on retry merges.     |
|               +--------------+---------------------------------------+
|                              |
|                              v
|               [ Evaluate Node (LLM-as-a-Judge) ]
|               | Fast-Path: If top score >= 0.82 -> Bypass Judge
|               | Otherwise: Pydantic GradeEvaluation (is_relevant, critique)
|               +--------------+---------------------------------------+
|                              |
|          (Relevant / Max Retries)         (Insufficient Context)
|                              |                       |
|                              v                       |
|               [ Assemble Multimodal Context ]        |
|               | Deduplicates parent context blocks.  |
|               | Concurrently loads & encodes images. |
|               | Builds inline citations [^N].        |
|               +--------------+-----------------------+
|                              |
|                              v
|               [ Generate Node ]
|               | Multimodal LLM synthesis with citations.
|               +--------------+
|                              |
|                              v
+-------------- [ Verify Groundedness Node ] (Self-Correction)
 (Unhallucinated) | Pydantic GroundednessEvaluation (is_grounded, score)
    Loopback      +--------------+
                                 | (Verified Grounded / Max Retries)
                                 v
                             [ [END] ] -> Final Markdown Response with Footnotes
```

---

## 2. Core Architectural Principles

### A. Single-Database Parent Payloads
Traditional hierarchical retrieval pairs a vector database (for child chunks) with an external document or key-value store (for parent sections). 

**RAG-Ultra's Solution:**
Child chunks are embedded in Chroma while the full parent page Markdown, image URI, and provenance metadata are stored **directly inside the child document's metadata payload**. This eliminates dual-database synchronization issues, network hops, and lookup latency.

### B. Anthropic Contextual Retrieval
Isolated chunks often lack the context needed for accurate semantic search. RAG-Ultra enriches every chunk with a page-level contextual overlay:
```text
[Context: Context from 'Technical Manual' (Page 2): Turbine Alpha-9 Wet Operating Protocols]
Pre-Operation Wet Checklist:
1. Verify IP67 waterproof enclosure seals...
```
This increases vector search recall on ambiguous queries by up to 35%.

### C. Corrective RAG (CRAG) with Structured Pydantic Output
Instead of simple linear retrieve-and-generate, the query execution graph verifies context before generating answers:
1. **High-Confidence Fast-Path**: When the top retrieved chunk has a high similarity score ($\ge 0.82$), the system directly proceeds to context assembly, bypassing the evaluator LLM to save tokens and ~40% latency.
2. **Structured LLM-as-a-Judge**: Evaluates context using Pydantic `GradeEvaluation`:
   ```python
   class GradeEvaluation(BaseModel):
       is_relevant: bool
       critique: str
       expanded_query: Optional[str] = None
   ```
3. **Query Expansion & Reciprocal Rank Fusion (RRF)**: On retries, the judge rewrites the query with synonyms. Newly retrieved chunks are merged with existing chunks using RRF to avoid discarding valid prior hits:
   $$\text{RRF Score}(d) = \sum_{r \in \text{runs}} \frac{1}{60 + \text{rank}(d, r)}$$

### D. Multimodal Context Assembly
- If retrieved parent metadata indicates `has_visuals = True` and contains an image path, the system concurrently fetches or reads the local JPEG image, base64 encodes it, and passes it as high-detail visual tokens to the multimodal generation model.
- If no graphics exist, the system passes only the structured Markdown to save token costs and latency.
- Generates structured inline footnotes (`[^1]`, `[^2]`) mapped to source documents and page numbers.

### E. Groundedness Verification (Self-Correction)
After generation, the `verify` node inspects the generated answer against the retrieved context:
```python
class GroundednessEvaluation(BaseModel):
    is_grounded: bool
    groundedness_score: float  # 0.0 - 1.0
    critique: str
```
If unsupported claims are detected and retries remain, the agent loops back to retrieval with a corrective query targeting the missing evidence.
