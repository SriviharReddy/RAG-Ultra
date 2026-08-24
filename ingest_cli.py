import os
import io
import argparse
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from my_agent.utils.tools import vision_ocr_parse
from core.database import get_database
from core.contextualizer import ContextualRetrievalEnricher
from core.config import get_settings

load_dotenv()

def render_and_cache_pdf_pages(pdf_path: str, doc_id: str, dpi: int = 150) -> List[Tuple[int, str, str, bool]]:
    """
    Renders each page of a PDF as a normalized JPEG image cached locally.
    Extracts native text, checks for visual graphics/tables, and returns:
    List of (page_num, image_rel_path, native_text, has_visuals).
    """
    settings = get_settings()
    doc_image_dir = os.path.join(settings.image_storage_dir, doc_id)
    os.makedirs(doc_image_dir, exist_ok=True)

    page_records = []
    pdf_document = pymupdf.open(pdf_path)
    total_pages = len(pdf_document)
    print(f"[Ingestion] Rendering '{pdf_path}' ({total_pages} pages) to local image storage...")

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = pdf_document.load_page(page_idx)

        # 1. Render image
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        image_filename = f"page_{page_num}.jpg"
        image_disk_path = os.path.join(doc_image_dir, image_filename)
        img.save(image_disk_path, format="JPEG", quality=85)
        
        # Static URL path for API clients/browser
        image_rel_url = f"/static/images/{doc_id}/{image_filename}"

        # 2. Extract native text layout
        native_text = page.get_text("text").strip()

        # 3. Detect visual components (embedded raster images, vector drawings, tables)
        embedded_images = page.get_images()
        drawings = page.get_drawings()
        has_visuals = (
            len(embedded_images) > 0
            or len(drawings) > 2
            or "table" in native_text.lower()
            or "|" in native_text
            or "figure" in native_text.lower()
            or "chart" in native_text.lower()
        )

        page_records.append((page_num, image_rel_url, image_disk_path, native_text, has_visuals))

    pdf_document.close()
    return page_records

def process_markdown_or_text_file(file_path: str, doc_id: str) -> List[Tuple[int, str, str, str, bool]]:
    """Splits plain Markdown/Text file into logical page/section blocks."""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split on explicit markdown page breaks or headers if available
    if "\n---\n" in full_text:
        pages = full_text.split("\n---\n")
    elif "\n# " in full_text:
        raw_sections = full_text.split("\n# ")
        pages = [("# " + s).strip() if idx > 0 else s.strip() for idx, s in enumerate(raw_sections) if s.strip()]
    else:
        pages = [full_text]

    page_records = []
    for idx, page_text in enumerate(pages):
        page_num = idx + 1
        has_visuals = "|" in page_text or "![" in page_text or "Table" in page_text or "Chart" in page_text
        page_records.append((page_num, "", "", page_text.strip(), has_visuals))

    return page_records

async def ingest_file(
    file_path: str,
    document_id: str = "doc_001",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Layout-aware ingestion pipeline:
    1. Page normalization & local image caching
    2. DeepSeek OCR with zero-crash native PyMuPDF fallback
    3. Anthropic-style Contextual Retrieval prefix generation
    4. Recursive Markdown-aware chunking preserving tables and headers
    5. Single-database parent payload indexing into Chroma
    """
    settings = get_settings()
    db = get_database()
    enricher = ContextualRetrievalEnricher()

    effective_chunk_size = chunk_size or settings.chunk_size
    effective_chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        keep_separator=True
    )

    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    doc_summary = f"Parsed technical document: '{filename}' (ID: {document_id})."

    print("\n========================================================")
    print(f"[Ingest Engine] Starting ingestion for '{filename}'...")
    print("========================================================")

    if file_ext == ".pdf":
        page_data = render_and_cache_pdf_pages(file_path, document_id)
    else:
        page_data = process_markdown_or_text_file(file_path, document_id)

    total_chunks_indexed = 0

    for record in page_data:
        page_num = record[0]
        image_rel_url = record[1]
        image_disk_path = record[2]
        native_text = record[3]
        has_visuals = record[4]

        print(f"\n--- Ingesting Page {page_num}/{len(page_data)} ---")

        # Step 2: Attempt OCR if local disk image is available, else native text fallback
        page_markdown = ""
        if image_disk_path and os.path.exists(image_disk_path):
            print(f"[Ingest Engine] Trying Vision OCR for page {page_num}...")
            ocr_result = await vision_ocr_parse.ainvoke({"image_source": image_disk_path})
            if ocr_result and len(ocr_result.strip()) > 20:
                page_markdown = ocr_result.strip()
                print(f"[Ingest Engine] Vision OCR succeeded ({len(page_markdown)} chars).")

        if not page_markdown:
            print(f"[Ingest Engine] Using native layout text ({len(native_text)} chars).")
            page_markdown = native_text or f"Page {page_num} content from {filename}."

        # Step 3: Contextual Retrieval prefix
        print("[Ingest Engine] Generating Contextual prefix...")
        context_prefix = await enricher.generate_page_prefix(doc_summary, page_markdown[:1500])
        print(f"[Ingest Engine] Prefix: \"{context_prefix}\"")

        # Step 4: Recursive Markdown chunking
        child_chunks = splitter.split_text(page_markdown)
        if not child_chunks:
            child_chunks = [page_markdown]

        print(f"[Ingest Engine] Split into {len(child_chunks)} layout-aware chunks.")

        # Step 5: Hierarchical indexing with parent payload
        db.ingest_hierarchical_document(
            parent_text=page_markdown,
            child_chunks=child_chunks,
            context_prefix=context_prefix,
            image_url=image_rel_url or image_disk_path,
            has_visuals=has_visuals,
            metadata_origin={
                "source": filename,
                "page": page_num,
                "doc_id": document_id,
                "image_disk_path": image_disk_path or ""
            }
        )
        total_chunks_indexed += len(child_chunks)

    print(f"\n[Ingest Engine] Ingestion complete: {len(page_data)} pages, {total_chunks_indexed} chunks indexed.")
    return {
        "status": "success",
        "doc_id": document_id,
        "source": filename,
        "pages_processed": len(page_data),
        "total_chunks_indexed": total_chunks_indexed
    }

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF or Markdown documents into the SOTA RAG database.")
    parser.add_argument("--pdf", "--file", dest="file_path", type=str, required=True, help="Path to local PDF/MD file")
    parser.add_argument("--id", dest="doc_id", type=str, default="doc_001", help="Document Identifier")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap for text splitting")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        exit(1)

    asyncio.run(ingest_file(
        file_path=args.file_path,
        document_id=args.doc_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    ))

if __name__ == "__main__":
    main()
