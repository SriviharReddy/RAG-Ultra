import os
import base64
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

def get_vector_db() -> Chroma:
    persist_dir = os.getenv("PERSIST_DIR", "./db_storage/chroma")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="sota_rag_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

@tool
def vector_search_db(query: str, k: int = 3) -> list:
    """Search the vector database for relevant chunks and return parent payloads."""
    db = get_vector_db()
    results = db.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        entry = {"parent_content": doc.metadata.get("parent_content", doc.page_content), **doc.metadata, "score": score}
        output.append(entry)
    return output

@tool
async def deepseek_ocr_parse(image_url: str) -> str:
    """Uses DeepSeek vision model to OCR and convert a page image to structured Markdown."""
    ocr_llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    message = HumanMessage(content=[
        {"type": "text", "text": "Convert this document page to complete, structured Markdown. Preserve all tables, headings, and lists accurately. Output only the Markdown."},
        {"type": "image_url", "image_url": {"url": image_url}}
    ])
    response = await ocr_llm.ainvoke([message])
    return response.content.strip()