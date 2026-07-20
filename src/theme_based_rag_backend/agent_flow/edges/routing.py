from src.theme_based_rag_backend.agent_flow.state import AgentState

def route_by_category(state: AgentState) -> str:
    return state["category"]

def route_after_critique(state: AgentState) -> str:
    feedback = state.get("critique_feedback")
    attempts = state.get("attempts", 0)
    
    if feedback == "PASS" or attempts >= 3:
        if attempts >= 3 and feedback != "PASS":
            print("Max refinement attempts reached. Proceeding with best effort response.")
        return "approved"
    return "rejected"
