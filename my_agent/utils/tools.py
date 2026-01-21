import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

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
    """Search the vector database for relevant chunks."""
    db = get_vector_db()
    results = db.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "score": score,
            **doc.metadata
        })
    return output