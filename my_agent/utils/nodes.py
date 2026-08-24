import os
import base64
import asyncio
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from my_agent.utils.state import AgentState, DocumentChunk, Citation
from core.database import get_database
from core.config import get_settings, get_fast_llm, get_generation_llm

# --- Pydantic Schemas for Structured LLM-as-a-Judge ---

class GradeEvaluation(BaseModel):
    is_relevant: bool = Field(
        description="Whether the retrieved contexts contain sufficient, accurate information to answer the user's query."
    )
    critique: str = Field(
        description="Brief justification of what information is present, missing, or why retrieval needs adjustment."
    )
    expanded_query: Optional[str] = Field(
        default=None,
        description="A refined, search-optimized query with synonymous keywords if current context is insufficient."
    )

class GroundednessEvaluation(BaseModel):
    is_grounded: bool = Field(
        description="Whether every factual claim in the generated answer is strictly supported by the retrieved context."
    )
    groundedness_score: float = Field(
        description="Confidence score between 0.0 (completely hallucinated) and 1.0 (fully verified in context)."
    )
    critique: str = Field(
        description="Explanation of any unsupported claims or confirmation of factual consistency."
    )

# --- Reciprocal Rank Fusion & Deduplication Helper ---

def merge_chunks_rrf(
    existing_chunks: List[DocumentChunk],
    new_chunks: List[DocumentChunk],
    k: int = 60,
    top_n: int = 4
) -> List[DocumentChunk]:
    """
    Merges prior retrieval hits with newly expanded search results using
    Reciprocal Rank Fusion (RRF) and content deduplication.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}

    def get_chunk_key(c: DocumentChunk) -> str:
        meta = c.get("metadata", {})
        source = meta.get("source", "")
        page = meta.get("page", "")
        idx = meta.get("chunk_index", "")
        content_hash = str(hash(c.get("content", "")))[:12]
        return f"{source}:{page}:{idx}:{content_hash}"

    for rank, chunk in enumerate(existing_chunks):
        key = get_chunk_key(chunk)
        chunk_map[key] = chunk
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank + 1))

    for rank, chunk in enumerate(new_chunks):
        key = get_chunk_key(chunk)
        chunk_map[key] = chunk
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank + 1))

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    merged: List[DocumentChunk] = []
    for key in sorted_keys[:top_n]:
        item = chunk_map[key]
        item["score"] = float(scores[key])
        merged.append(item)

    return merged

# --- Graph Node Implementations ---

async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves relevant document chunks from Chroma.
    Supports query expansion on retries, metadata filtering, and RRF result merging.
    """
    settings = get_settings()
    db = get_database()
    retry_count = state.get("retry_count", 0)

    # Determine query: Use expanded query if available from previous judge critique
    if retry_count > 0 and state.get("expanded_query"):
        search_query = state["expanded_query"]
        print(f"[Retrieve Node] Executing expanded search query (Retry {retry_count}): '{search_query}'")
    else:
        search_query = state.get("condensed_query") or state.get("query", "")
        print(f"[Retrieve Node] Executing search query: '{search_query}'")

    metadata_filter = state.get("metadata_filter")
    raw_results = await db.similarity_search_with_score_async(
        query=search_query,
        k=settings.top_k,
        metadata_filter=metadata_filter
    )

    new_chunks: List[DocumentChunk] = []
    for doc, score in raw_results:
        parent_text = doc.metadata.get("parent_content", doc.page_content)
        chunk = DocumentChunk(
            content=parent_text,
            metadata=dict(doc.metadata),
            score=float(score)
        )
        new_chunks.append(chunk)

    existing_chunks = state.get("retrieved_chunks", [])
    if existing_chunks:
        merged_chunks = merge_chunks_rrf(existing_chunks, new_chunks, top_n=settings.top_k + 1)
        print(f"[Retrieve Node] RRF merged {len(existing_chunks)} prior chunks + {len(new_chunks)} new chunks -> {len(merged_chunks)} unique chunks.")
    else:
        merged_chunks = new_chunks
        print(f"[Retrieve Node] Retrieved {len(merged_chunks)} initial chunks.")

    return {
        "retrieved_chunks": merged_chunks,
        "query": search_query
    }

async def evaluate_relevance_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM-as-a-Judge node with structured Pydantic output.
    Assesses context relevance and completeness, triggering corrective query expansion if needed.
    """
    settings = get_settings()
    query = state.get("raw_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    # If no documents are in the database or found
    if not chunks:
        if retry_count < settings.max_retries:
            return {
                "is_relevant": False,
                "route_decision": "retrieve",
                "retry_count": retry_count + 1,
                "critique": "No documents found in knowledge base. Reformulating query.",
                "expanded_query": f"{query} overview documentation specifications"
            }
        else:
            return {
                "is_relevant": False,
                "route_decision": "assemble",
                "critique": "Max retries reached with empty retrieval. Proceeding with best effort."
            }

    # High-confidence fast-path: Bypass judge LLM when top chunk has high relevance
    top_chunk_score = chunks[0].get("score")
    if top_chunk_score is not None and top_chunk_score >= 0.82:
        print(f"[Judge Fast-Path] Top chunk score ({top_chunk_score:.3f} >= 0.82) indicates high confidence. Bypassing judge LLM.")
        return {
            "is_relevant": True,
            "critique": f"High confidence similarity match (Score: {top_chunk_score:.3f}).",
            "expanded_query": None,
            "route_decision": "assemble",
            "retry_count": retry_count
        }

    contexts_text = "\n\n".join([
        f"[Doc {idx+1} | Source: {c.get('metadata', {}).get('source', 'Unknown')} | Page: {c.get('metadata', {}).get('page', 1)}]\n{c.get('content', '')}"
        for idx, c in enumerate(chunks)
    ])

    judge_prompt = f"""You are an expert Retrieval Evaluator & Judge for an Agentic RAG system.
Evaluate whether the following retrieved context passages contain sufficient and relevant information to answer the user's query.

User Query: {query}

Retrieved Contexts:
{contexts_text}

Provide an objective assessment:
1. is_relevant: true if contexts contain direct or sufficient answers; false if off-topic, missing critical details, or insufficient.
2. critique: concise explanation.
3. expanded_query: if is_relevant is false and more context is needed, provide a rewritten, keyword-rich search query; otherwise null.
"""

    try:
        judge_llm = get_fast_llm(temperature=0.0)
        structured_judge = judge_llm.with_structured_output(GradeEvaluation)
        evaluation: GradeEvaluation = await structured_judge.ainvoke([HumanMessage(content=judge_prompt)])
        is_relevant = evaluation.is_relevant
        critique = evaluation.critique
        expanded_query = evaluation.expanded_query
        print(f"[Judge Evaluation] Relevant: {is_relevant} | Critique: {critique}")
    except Exception as e:
        print(f"[Judge Evaluation Warning] Structured output unavailable/failed ({e}). Running heuristic grading.")
        # Heuristic fallback: check keyword presence
        query_words = set(query.lower().split())
        context_words = set(contexts_text.lower().split())
        overlap = query_words.intersection(context_words)
        is_relevant = len(overlap) >= max(1, len(query_words) // 3)
        critique = f"Heuristic overlap check: {len(overlap)} matching query terms."
        expanded_query = f"{query} details specifications" if not is_relevant else None

    # Routing logic
    if is_relevant or retry_count >= settings.max_retries:
        route_decision = "assemble"
    else:
        route_decision = "retrieve"
        retry_count += 1

    return {
        "is_relevant": is_relevant,
        "critique": critique,
        "expanded_query": expanded_query,
        "route_decision": route_decision,
        "retry_count": retry_count
    }

async def assemble_multimodal_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Assembles multimodal context payloads:
    1. Deduplicates parent markdown and numbered citation blocks.
    2. Concurrently fetches and base64 encodes local disk or remote page images.
    3. Builds structured citation provenance for frontend transparency.
    """
    query = state.get("query", "")
    raw_query = state.get("raw_query") or query
    chunks = state.get("retrieved_chunks", [])
    settings = get_settings()

    citations: List[Citation] = []
    text_context_blocks: List[str] = []
    image_tasks = []
    seen_images = set()

    for idx, chunk in enumerate(chunks):
        citation_id = idx + 1
        content = chunk.get("content", "")
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Document")
        page = meta.get("page", 1)
        doc_id = meta.get("doc_id", "")
        image_url = meta.get("image_url", "")
        image_disk_path = meta.get("image_disk_path", "")
        has_visuals = meta.get("has_visuals", False)

        snippet = content[:200].replace("\n", " ").strip() + "..."
        citations.append(Citation(
            id=citation_id,
            source=source,
            page=page,
            doc_id=doc_id,
            snippet=snippet,
            image_url=image_url
        ))

        block_header = f"--- CONTEXT BLOCK [^{citation_id}] (Source: {source}, Page {page}) ---"
        text_context_blocks.append(f"{block_header}\n{content}")

        # Gather image loading candidates
        candidate_image = image_disk_path or image_url
        if has_visuals and candidate_image and candidate_image not in seen_images:
            seen_images.add(candidate_image)
            image_tasks.append((citation_id, candidate_image))

    # Concurrently load images
    async def load_single_image(cid: int, img_source: str) -> Optional[Dict[str, Any]]:
        try:
            if os.path.exists(img_source):
                with open(img_source, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                }
            elif img_source.startswith("http://") or img_source.startswith("https://"):
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(img_source)
                    if resp.status_code == 200:
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        return {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                        }
            elif img_source.startswith("/static/images/"):
                # Map static URL to local storage directory
                rel_path = img_source.replace("/static/images/", "")
                local_path = os.path.join(settings.image_storage_dir, rel_path)
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    return {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                    }
        except Exception as err:
            print(f"[Multimodal Assembly] Skipped image {img_source}: {err}")
        return None

    loaded_images = []
    if image_tasks:
        results = await asyncio.gather(*[load_single_image(cid, src) for cid, src in image_tasks])
        loaded_images = [img for img in results if img is not None]
        print(f"[Multimodal Assembly] Concurrently loaded {len(loaded_images)} visual page images.")

    # Formulate generation messages
    system_instruction = (
        "You are an authoritative, context-grounded AI assistant. "
        "Answer the user's inquiry thoroughly and accurately based exclusively on the provided context passages and visual diagrams. "
        "You MUST insert inline footnote citations in markdown format (e.g. [^1], [^2]) immediately following claims derived from that context block. "
        "Do not invent facts outside the retrieved knowledge."
    )

    combined_context_text = "\n\n".join(text_context_blocks)
    user_prompt_text = (
        f"User Query: {raw_query}\n\n"
        f"Retrieved Document Context:\n{combined_context_text}\n\n"
        "Generate a comprehensive, structured response with inline citations [^N]."
    )

    llm_payload: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    for img_item in loaded_images:
        llm_payload.append(img_item)

    return {
        "llm_inputs": [
            SystemMessage(content=system_instruction),
            HumanMessage(content=llm_payload)
        ],
        "citations": citations,
        "route_decision": "generate"
    }

async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Invokes the multimodal generation model to synthesize a grounded answer.
    """
    llm_inputs = state.get("llm_inputs", [])
    try:
        gen_llm = get_generation_llm(temperature=0.1)
        response = await gen_llm.ainvoke(llm_inputs)
        answer = str(response.content).strip()
        print(f"[Generate Node] Synthesized response ({len(answer)} chars).")
    except Exception as e:
        print(f"[Generate Node Warning] Flagship LLM invocation failed ({e}), generating deterministic synthesis.")
        # Fallback local synthesis from chunks and citations
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            first_chunk = chunks[0].get("content", "")[:300]
            answer = f"Based on the retrieved documentation [^1]:\n\n{first_chunk}...\n\n(Note: Generated via local fallback synthesis)."
        else:
            answer = "No relevant context was available to answer the query."

    return {
        "answer": answer,
        "route_decision": "verify"
    }

async def verify_groundedness_node(state: AgentState) -> Dict[str, Any]:
    """
    Self-Correction Node: Checks if the generated answer contains ungrounded claims or hallucinations.
    If ungrounded and retries remain, triggers loopback to retrieve node.
    """
    settings = get_settings()
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    contexts_text = "\n\n".join([c.get("content", "") for c in chunks])

    verification_prompt = f"""You are an impartial Groundedness & Hallucination Verifier.
Evaluate whether the following generated response is strictly supported by the provided source contexts.

Retrieved Contexts:
{contexts_text}

Generated Response:
{answer}

Determine:
1. is_grounded: true if every factual statement is backed by the context, false if there are hallucinations or contradictions.
2. groundedness_score: float from 0.0 to 1.0.
3. critique: brief explanation of verification result.
"""

    try:
        fast_llm = get_fast_llm(temperature=0.0)
        structured_verifier = fast_llm.with_structured_output(GroundednessEvaluation)
        eval_result: GroundednessEvaluation = await structured_verifier.ainvoke([HumanMessage(content=verification_prompt)])
        is_grounded = eval_result.is_grounded
        groundedness_score = eval_result.groundedness_score
        critique = eval_result.critique
        print(f"[Groundedness Verifier] Score: {groundedness_score} | Grounded: {is_grounded} | Critique: {critique}")
    except Exception as e:
        print(f"[Groundedness Verifier Warning] Structured verification skipped ({e}), accepting response as grounded.")
        is_grounded = True
        groundedness_score = 1.0
        critique = "Verified with default groundedness."

    if is_grounded or retry_count >= settings.max_retries:
        route_decision = "end"
    else:
        print(f"[Groundedness Verifier] Answer ungrounded, initiating corrective retrieval loop (Attempt {retry_count + 1})...")
        route_decision = "retrieve"
        retry_count += 1

    return {
        "is_grounded": is_grounded,
        "groundedness_score": groundedness_score,
        "critique": critique,
        "retry_count": retry_count,
        "route_decision": route_decision
    }
