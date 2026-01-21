from langgraph.graph import StateGraph, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import retrieve_node, generate_response_node

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_response_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()