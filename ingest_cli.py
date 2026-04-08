# ingest_cli.py
import os
import argparse
import asyncio
import io
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
from my_agent.utils.tools import deepseek_ocr_parse
from core.database import SotaRagDatabase
from core.contextualizer import ContextualRetrievalEnricher

load_dotenv()

def normalize_pdf_to_images(pdf_path: str, dpi: int = 150) -> list[bytes]:
    """Converts a multi-page PDF into normalized image bytes at target DPI."""
    images_bytes = []
    pdf_document = fitz.open(pdf_path)
    print(f"Normalizing '{pdf_path}' ({len(pdf_document)} pages) to images...")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        buffer = io.BytesIO()
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img.save(buffer, format="JPEG", quality=85)
        images_bytes.append(buffer.getvalue())
    pdf_document.close()
    return images_bytes

async def ingest_document(pdf_path: str, document_id: str):
    """
    Full ingestion pipeline:
    1. PyMuPDF normalizer
    2. DeepSeek-OCR tool calls
    3. Contextual summary prefixing
    4. Hierarchical parent-payload storage in Chroma
    """
    db_wrapper = SotaRagDatabase()
    enricher = ContextualRetrievalEnricher()
    pages_bytes = normalize_pdf_to_images(pdf_path)
    doc_summary = f"This document represents the parsed handbook: '{os.path.basename(pdf_path)}'."
    print("\nStarting Ingestion Pipeline...")
    for idx, page_data in enumerate(pages_bytes):
        page_num = idx + 1
        print(f"\n--- Processing Page {page_num}/{len(pages_bytes)} ---")
        mock_hosted_url = f"https://my-bucket.s3.amazonaws.com/docs/{document_id}/page_{page_num}.jpg"
        print("Calling DeepSeek-OCR API tool...")
        ocr_markdown = await deepseek_ocr_parse.ainvoke({"image_url": mock_hosted_url})
        has_visuals = "Table" in ocr_markdown or "Figure" in ocr_markdown or "Chart" in ocr_markdown
        print("Generating Contextual Retrieval prefix...")
        context_prefix = await enricher.generate_page_prefix(doc_summary, ocr_markdown[:1500])
        print(f"Context Prefix: \"{context_prefix}\"")
        parent_text = ocr_markdown
        chunk_size = 400
        overlap = 50
        child_chunks = []
        i = 0
        while i < len(parent_text):
            child_chunks.append(parent_text[i:i+chunk_size])
            i += (chunk_size - overlap)
        print(f"Indexing {len(child_chunks)} child chunks...")
        db_wrapper.ingest_hierarchical_document(
            parent_text=parent_text,
            child_chunks=child_chunks,
            context_prefix=context_prefix,
            image_url=mock_hosted_url,
            has_visuals=has_visuals,
            metadata_origin={"source": os.path.basename(pdf_path), "page": page_num, "doc_id": document_id}
        )
    print(f"\nSuccessfully completed ingestion of '{pdf_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the SOTA RAG database using DeepSeek-OCR.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to local PDF file")
    parser.add_argument("--id", type=str, default="doc_001", help="Document Identifier")
    args = parser.parse_args()
    if not os.path.exists(args.pdf):
        print(f"Error: File '{args.pdf}' does not exist.")
        exit(1)
    asyncio.run(ingest_document(args.pdf, args.id))