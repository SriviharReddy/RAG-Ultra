# ingest_cli.py
import os
import argparse
import asyncio
import io
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
from core.database import SotaRagDatabase

load_dotenv()

def normalize_pdf_to_images(pdf_path: str, dpi: int = 150) -> list[bytes]:
    """Converts each PDF page to a JPEG image in memory."""
    images_bytes = []
    pdf_document = fitz.open(pdf_path)
    print(f"Normalizing '{pdf_path}' ({len(pdf_document)} pages)...")
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
    db_wrapper = SotaRagDatabase()
    pages_bytes = normalize_pdf_to_images(pdf_path)
    for idx, page_data in enumerate(pages_bytes):
        page_num = idx + 1
        # Placeholder: real OCR would go here
        placeholder_text = f"[Page {page_num} of {os.path.basename(pdf_path)}]"
        db_wrapper.ingest_document(
            text=placeholder_text,
            metadata={"source": os.path.basename(pdf_path), "page": page_num, "doc_id": document_id}
        )
        print(f"Indexed page {page_num}")
    print(f"\nIngestion complete: '{pdf_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the RAG database.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to local PDF file")
    parser.add_argument("--id", type=str, default="doc_001", help="Document Identifier")
    args = parser.parse_args()
    if not os.path.exists(args.pdf):
        print(f"Error: File '{args.pdf}' does not exist.")
        exit(1)
    asyncio.run(ingest_document(args.pdf, args.id))