import base64
import os
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from my_agent.utils.state import AgentState
from my_agent.utils.tools import vector_search_db

load_dotenv()

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    search_query = query
    if retry_count > 0:
        expansion_prompt = f"Given the query: '{query}', write an expanded semantic search query seeking deep context."
        expanded_res = await judge_llm.ainvoke([HumanMessage(content=expansion_prompt)])
        search_query = expanded_res.content.strip()
    results = vector_search_db.invoke({"query": search_query, "k": 3})
    from my_agent.utils.state import DocumentChunk
    chunks = []
    if isinstance(results, list) and results and "error" not in results[0]:
        for res in results:
            chunk = DocumentChunk(
                content=res["parent_content"],
                metadata={"has_visuals": res.get("has_visuals"), "image_url": res.get("image_url"), "page": res.get("page"), "source": res.get("source")}
            )
            chunks.append(chunk)
    return {"retrieved_chunks": chunks, "retry_count": retry_count + 1}

async def evaluate_relevance_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)
    if not chunks:
        return {"response": "generate" if retry_count >= 3 else "retrieve"}
    contexts = [c["content"] for c in chunks]
    prompt = f"""Analyze if the retrieved context is sufficient to answer the query.
Query: {query}
Contexts: {' '.join(contexts)}
Reply ONLY with 'YES' or 'NO'."""
    judge_response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    decision = "generate"
    if "NO" in judge_response.content.strip().upper() and retry_count < 3:
        decision = "retrieve"
    return {"response": decision}

async def assemble_multimodal_context_node(state: AgentState) -> Dict[str, Any]:
    """Assembles text + optional base64-encoded page images for the VLM."""
    query = state["query"]
    chunks = state["retrieved_chunks"]
    message_contents = [{"type": "text", "text": f"User Query: {query}\n\nAnswer strictly based on the retrieved context below."}]
    async with httpx.AsyncClient() as client:
        for idx, chunk in enumerate(chunks):
            text_content = chunk["content"]
            metadata = chunk["metadata"]
            message_contents.append({"type": "text", "text": f"--- CONTEXT {idx+1} (Source: {metadata.get('source')}, Page {metadata.get('page')}) ---\n{text_content}"})
            if metadata.get("has_visuals") and metadata.get("image_url"):
                try:
                    img_response = await client.get(metadata["image_url"])
                    if img_response.status_code == 200:
                        b64_image = base64.b64encode(img_response.content).decode("utf-8")
                        message_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"}})
                except Exception as e:
                    print(f"Image download failed: {e}")
    return {"llm_inputs": message_contents}

async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    message = HumanMessage(content=state["llm_inputs"])
    response = await generation_llm.ainvoke([message])
    return {"response": response.content}