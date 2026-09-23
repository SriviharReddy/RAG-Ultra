import logging

logger = logging.getLogger(__name__)

import os
import base64
import mimetypes
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.config import get_settings

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
            mime_type = mimetypes.guess_type(image_source)[0] or "image/jpeg"
            with open(image_source, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            image_url_payload = {"url": f"data:{mime_type};base64,{b64_data}"}
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
        logger.warning(f"[OCR Tool] {provider_name} call failed ({e}), falling back to native extractor.")
        return ""

