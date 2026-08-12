from langgraph.graph import StateGraph, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import (
    retrieve_node,
    evaluate_relevance_node,
    assemble_multimodal_context_node,
    generate_response_node,
    verify_groundedness_node
)

# Initialize LangGraph StateGraph with typed AgentState
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_relevance_node)
workflow.add_node("assemble", assemble_multimodal_context_node)
workflow.add_node("generate", generate_response_node)
workflow.add_node("verify", verify_groundedness_node)

# Set entry point
workflow.set_entry_point("retrieve")

# Edges
workflow.add_edge("retrieve", "evaluate")

def route_relevance(state: AgentState) -> str:
    """Routes based on LLM-as-a-Judge relevance determination."""
    decision = state.get("route_decision", "assemble")
    if decision == "retrieve":
        return "retrieve"
    return "assemble"

workflow.add_conditional_edges(
    "evaluate",
    route_relevance,
    {
        "retrieve": "retrieve",
        "assemble": "assemble"
    }
)

workflow.add_edge("assemble", "generate")
workflow.add_edge("generate", "verify")

def route_groundedness(state: AgentState) -> str:
    """Routes based on groundedness self-correction verification."""
    decision = state.get("route_decision", "end")
    if decision == "retrieve":
        return "retrieve"
    return END

workflow.add_conditional_edges(
    "verify",
    route_groundedness,
    {
        "retrieve": "retrieve",
        END: END
    }
)

# Compile graph
graph = workflow.compile()
