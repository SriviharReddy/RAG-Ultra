# core/contextualizer.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class ContextualRetrievalEnricher:
    def __init__(self):
        # Uses low-cost, low-latency 2026 standard model
        self.llm = ChatOpenAI(model="gpt-5.5-instant", temperature=0)

    async def generate_page_prefix(self, document_summary: str, page_content: str) -> str:
        """Generates a concise 1-sentence contextual overlay for a chunk."""
        prompt = f"""
        Given the following document summary and page content, write a single-sentence context prefix.
        This prefix will be prepended to search chunks from this page to make them self-contained.
        
        Document Summary: {document_summary}
        Page Content: {page_content}
        
        Answer ONLY with the single-sentence prefix. Do not add introductions, quotes, or markdown wrappers.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()