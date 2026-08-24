import os
import base64
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.database import get_database
from core.config import get_settings

@tool
def vector_search_db(query: str, k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search the vector database for relevant chunks and return parent payloads with scores."""
    db = get_database()
    results = db.similarity_search_with_score(query, k=k, metadata_filter=metadata_filter)
    output = []
    for doc, score in results:
        entry = {
            "parent_content": doc.metadata.get("parent_content", doc.page_content),
            "score": float(score),
            **doc.metadata
        }
        output.append(entry)
    return output

@tool
async def vector_search_db_async(query: str, k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Asynchronously search the vector database for relevant chunks."""
    db = get_database()
    results = await db.similarity_search_with_score_async(query, k=k, metadata_filter=metadata_filter)
    output = []
    for doc, score in results:
        entry = {
            "parent_content": doc.metadata.get("parent_content", doc.page_content),
            "score": float(score),
            **doc.metadata
        }
        output.append(entry)
    return output

@tool
async def vision_ocr_parse(image_source: str) -> str:
    """
    Uses a Vision LLM (Novita AI, OpenAI Vision, or custom VLM endpoint) to OCR and convert
    a page image (URL or local path) to structured Markdown.
    Falls back to native extraction if no vision provider is configured or available.
    """
    settings = get_settings()

    # Determine active Vision/OCR provider
    ocr_client_kwargs: Dict[str, Any] = {}
    provider_name = ""

    if settings.novita_api_key:
        provider_name = "Novita AI"
        ocr_client_kwargs = {
            "model": settings.novita_model,
            "base_url": settings.novita_base_url,
            "api_key": settings.novita_api_key,
            "temperature": 0.0
        }
    elif settings.ocr_api_key and settings.ocr_base_url:
        provider_name = "Custom OCR"
        ocr_client_kwargs = {
            "model": settings.ocr_model,
            "base_url": settings.ocr_base_url,
            "api_key": settings.ocr_api_key,
            "temperature": 0.0
        }
    elif settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        provider_name = "OpenAI Vision"
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        ocr_client_kwargs = {
            "model": settings.ocr_model or "gpt-4o-mini",
            "api_key": api_key,
            "temperature": 0.0
        }
        if settings.openai_base_url:
            ocr_client_kwargs["base_url"] = settings.openai_base_url
    else:
        # No external vision API available; fallback to PyMuPDF native extractor
        return ""

    try:
        # Check if local image file or URL
        if os.path.exists(image_source):
            with open(image_source, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            image_url_payload = {"url": f"data:image/jpeg;base64,{b64_data}"}
        else:
            image_url_payload = {"url": image_source}

        ocr_llm = ChatOpenAI(**ocr_client_kwargs)
        message = HumanMessage(content=[
            {"type": "text", "text": "Convert this document page into clean, structured Markdown. Format all tables using standard GitHub-flavored Markdown (| col |). Render mathematical equations using LaTeX delimiters ($...$ for inline, $$...$$ for block). Preserve headings, footnotes, and bullet hierarchies. Output ONLY the raw Markdown."},
            {"type": "image_url", "image_url": image_url_payload}
        ])
        response = await ocr_llm.ainvoke([message])
        return str(response.content).strip()
    except Exception as e:
        print(f"[OCR Tool] {provider_name} call failed ({e}), falling back to native extractor.")
        return ""

# Backward compatibility alias
deepseek_ocr_parse = vision_ocr_parse
