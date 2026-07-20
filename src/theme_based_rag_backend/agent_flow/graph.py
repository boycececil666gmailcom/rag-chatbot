from langgraph.graph import StateGraph, END
from src.theme_based_rag_backend.agent_flow.state import AgentState
from src.theme_based_rag_backend.agent_flow.nodes import (
    classifier_node,
    rag_qa_node,
    safeguard_node,
    critique_node
)
from src.theme_based_rag_backend.agent_flow.edges import (
    route_by_category,
    route_after_critique
)

# Workflow Graph Setup
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classifier", classifier_node)
workflow.add_node("rag_qa", rag_qa_node)
workflow.add_node("safeguard", safeguard_node)
workflow.add_node("critique", critique_node)

# Set Entry Point and Edges
workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_by_category,
    {
        "rag": "rag_qa",
        "refuse": "safeguard"
    }
)

workflow.add_edge("rag_qa", "critique")
workflow.add_edge("safeguard", "critique")

workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "approved": END,
        "rejected": "classifier"  # Loop back to the start (Classifier Node)
    }
)

# Compile Workflow Graph
agent_graph = workflow.compile()
