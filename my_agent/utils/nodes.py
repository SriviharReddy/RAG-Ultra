import base64
import os
import httpx
from typing import Literal, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from my_agent.utils.state import AgentState, DocumentChunk
from my_agent.utils.tools import vector_search_db

# Load environment variables before instantiating models
load_dotenv()


# 1. Initialize modern 2026 models
# Low-latency reasoning model for evaluations
judge_llm = ChatOpenAI(model="gpt-5.5-instant", temperature=0)
# Flagship model for final multimodal generation
generation_llm = ChatOpenAI(model="gpt-5.5", temperature=0.1)

async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Performs similarity search on child chunks and extracts the parent markdown
    and image URLs directly from the metadata payloads.
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    
    # Simple query expansion on retries
    search_query = query
    if retry_count > 0:
        # Ask judge LLM to expand the query based on state query
        expansion_prompt = f"Given the query: '{query}', write an expanded semantic search query seeking deep context."
        expanded_res = await judge_llm.ainvoke([HumanMessage(content=expansion_prompt)])
        search_query = expanded_res.content.strip()

    # Call search tool directly (sync wrapper inside node)
    search_results = vector_search_db.invoke({"query": search_query, "k": 3})
    
    retrieved_chunks = []
    if isinstance(search_results, list) and len(search_results) > 0 and "error" not in search_results[0]:
        for res in search_results:
            chunk = DocumentChunk(
                content=res["parent_content"],
                metadata={
                    "has_visuals": res["has_visuals"],
                    "image_url": res["image_url"],
                    "page": res["page"],
                    "source": res["source"]
                }
            )
            retrieved_chunks.append(chunk)

    return {
        "retrieved_chunks": retrieved_chunks,
        "retry_count": retry_count + 1
    }

async def evaluate_relevance_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM-as-a-Judge node. Evaluates whether the retrieved parent contexts 
    are sufficient to solve the query. Returns a route decision.
    """
    query = state["query"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)
    
    if not chunks:
        # Bypasses to generate (or end) if no documents found after max retries
        return {"response": "generate" if retry_count >= 3 else "retrieve"}
        
    contexts = [c["content"] for c in chunks]
    prompt = f"""
    Analyze if the retrieved context is sufficient and relevant to answer the query.
    Query: {query}
    Contexts: {' '.join(contexts)}
    
    Reply ONLY with 'YES' if the contexts are highly relevant and sufficient, or 'NO' if they are insufficient.
    """
    
    judge_response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    response_text = judge_response.content.strip().upper()
    
    # Route decision saved in state or conditional edge resolver
    decision = "generate"
    if "NO" in response_text and retry_count < 3:
        decision = "retrieve"
        
    return {"response": decision}

async def assemble_multimodal_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Checks the unified payload metadata. If parent page contains complex graphics/charts,
    downloads and base64 encodes the page image to pass to the VLM.
    Otherwise, passes only the structured parent Markdown to save tokens and latency.
    """
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    message_contents = [
        {"type": "text", "text": f"User Query: {query}\n\nAnswer the query strictly based on the retrieved context below."}
    ]
    
    async with httpx.AsyncClient() as client:
        for idx, chunk in enumerate(chunks):
            text_content = chunk["content"]
            metadata = chunk["metadata"]
            
            # Inject parent markdown text
            message_contents.append({
                "type": "text",
                "text": f"--- RETRIEVED PAGE CONTEXT {idx+1} (Source: {metadata.get('source')}, Page {metadata.get('page')}) ---\n{text_content}"
            })
            
            # Conditional Trigger: If visuals exist, download and base64 encode parent page image
            if metadata.get("has_visuals") and metadata.get("image_url"):
                try:
                    img_response = await client.get(metadata["image_url"])
                    if img_response.status_code == 200:
                        b64_image = base64.b64encode(img_response.content).decode("utf-8")
                        message_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        })
                except Exception as e:
                    # Fail silently and fallback to text representation
                    print(f"Bypassed image download due to error: {e}")
                    
    return {"llm_inputs": message_contents}

async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Sends the conditionally assembled text + visual context to the final Multimodal model.
    """
    message = HumanMessage(content=state["llm_inputs"])
    response = await generation_llm.ainvoke([message])
    return {"response": response.content}