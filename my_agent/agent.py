from langgraph.graph import StateGraph, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import retrieve_node, evaluate_relevance_node, assemble_multimodal_context_node, generate_response_node

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_relevance_node)
workflow.add_node("assemble", assemble_multimodal_context_node)
workflow.add_node("generate", generate_response_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "evaluate")

def route_relevance(state: AgentState):
    decision = state.get("response", "assemble")
    if decision == "retrieve":
        return "retrieve"
    return "assemble"

workflow.add_conditional_edges("evaluate", route_relevance, {"retrieve": "retrieve", "assemble": "assemble"})
workflow.add_edge("assemble", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()