
from langchain_core.messages import HumanMessage
from core.config import get_fast_llm

class ContextualRetrievalEnricher:
    """
    Generates contextual prefixes (Anthropic Contextual Retrieval pattern)
    for individual document chunks to ensure high recall when searched in isolation.
    Includes graceful fallback for offline or unconfigured environments.
    """
    def __init__(self):
        try:
            self.llm = get_fast_llm(temperature=0.0)
        except Exception:
            self.llm = None

    async def generate_page_prefix(self, document_summary: str, page_content: str) -> str:
        """Generates a concise 1-sentence contextual overlay for a chunk."""
        prompt = f"""
Given the following document summary and page content, write a single-sentence context prefix.
This prefix will be prepended to search chunks from this page to make them self-contained.

Document Summary: {document_summary}
Page Content: {page_content}

Answer ONLY with the single-sentence prefix. Do not add introductions, quotes, or markdown wrappers.
"""
        if self.llm is not None:
            try:
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                prefix = response.content.strip()
                if prefix:
                    return prefix
            except Exception as e:
                # Gracefully fallback if LLM endpoint is unreachable
                print(f"[Contextualizer] LLM invocation skipped/failed ({e}), applying heuristic prefix.")

        # Fallback deterministic prefix
        clean_summary = document_summary.replace("\n", " ").strip()
        first_line = page_content.strip().split("\n")[0][:80].strip()
        return f"Context from '{clean_summary}': {first_line}"
