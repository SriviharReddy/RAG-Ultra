# core/contextualizer.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class ContextualRetrievalEnricher:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def generate_page_prefix(self, document_summary: str, page_content: str) -> str:
        """Generates a concise 1-sentence contextual overlay for a chunk."""
        prompt = f"""Given the document summary and page content, write a single-sentence context prefix.
This prefix will be prepended to search chunks to make them self-contained.

Document Summary: {document_summary}
Page Content: {page_content}

Answer ONLY with the single-sentence prefix."""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()