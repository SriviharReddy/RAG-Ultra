# RAG-Ultra

**RAG-Ultra** is a high-accuracy multimodal RAG (Retrieval-Augmented Generation) chatbot designed to handle complex documents including scanned PDFs, tables, formulas, and visual content. 

It leverages a robust tech stack to ensure reliability and detailed reasoning:
- **Amazon Textract**: For enterprise-grade OCR ingestion, capable of preserving layout and extracting tables/forms from scanned documents.
- **Advanced Retrieval**: Implements Parent-Document Retrieval to maintain context and reduce hallucinations.
- **Reranking**: integrated cross-encoder/reranking steps to ensure the most relevant chunks are sent to the LLM.
- **Multimodal capabilities**: Uses Vision-Language (VL) models for reasoning over chart/diagram extracts.
- **LangSmith**: Full observability integration for tracing and debugging.

## Architecture

1.  **Ingestion Pipeline**:
    *   PDFs are processed via `boto3` and **Amazon Textract**.
    *   Text, tables, and raw image regions are extracted.
    *   Data is chunked and indexed. Text goes to a vector store; images are summarized by a VLM (e.g., GPT-4o, Claude 3.5 Sonnet) and embedded, or indexed purely by their summaries.

2.  **Retrieval**:
    *   **Parent Document Retriever**: Fetches larger context blocks for retrieved small chunks.
    *   **Reranking**: Re-orders results based on relevance score to improve precision.

3.  **Generation (LangGraph)**:
    *   A directed cyclic graph (DAG) manages the state.
    *   Nodes: `Retrieve` -> `Grade Documents` -> `Rerank` -> `Generate`.
    *   Support for "Visual Reasoning" loops if image context is retrieved.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SriviharReddy/RAG-Ultra.git
    cd RAG-Ultra
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your keys:
    *   `OPENAI_API_KEY` (for embeddings/generation)
    *   `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for Textract)
    *   `LANGCHAIN_API_KEY` (for LangSmith)

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Tech Stack
*   **LangChain / LangGraph**
*   **Streamlit**
*   **Amazon Textract**
*   **ChromaDB** (Vector Store)
*   **LangSmith**

## License
MIT
