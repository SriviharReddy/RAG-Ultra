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
async def deepseek_ocr_parse(image_source: str) -> str:
    """
    Uses DeepSeek vision model to OCR and convert a page image (URL or local path) to structured Markdown.
    Gracefully falls back if API key is not configured.
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        return ""

    try:
        # Check if local image file or URL
        if os.path.exists(image_source):
            with open(image_source, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            image_url_payload = {"url": f"data:image/jpeg;base64,{b64_data}"}
        else:
            image_url_payload = {"url": image_source}

        ocr_llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key
        )
        message = HumanMessage(content=[
            {"type": "text", "text": "Convert this document page to complete, structured Markdown. Preserve all tables, headings, and lists accurately. Output only the Markdown."},
            {"type": "image_url", "image_url": image_url_payload}
        ])
        response = await ocr_llm.ainvoke([message])
        return str(response.content).strip()
    except Exception as e:
        print(f"[OCR Tool] DeepSeek OCR call failed ({e}), falling back to native extractor.")
        return ""
