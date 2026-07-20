import logging
from langchain_core.messages import SystemMessage
from src.theme_based_rag_backend.config import CHATBOT_THEME
from src.theme_based_rag_backend.agent_flow.state import AgentState

logger = logging.getLogger(__name__)

def critique_node(state: AgentState) -> dict:
    from src.theme_based_rag_backend.agent_flow import llm
    category = state.get("category")
    draft = state.get("draft_response")
    docs = state.get("retrieved_documents")
    query = state["message"]
    attempts = state.get("attempts", 0)
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Critique Node validating response ({category})\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    if category == "refuse":
        critique_prompt = (
            f"You are a strict quality control evaluator.\n"
            f"Your task is to verify if the draft response is a polite refusal to answer a query outside the theme: '{CHATBOT_THEME}'.\n"
            f"Make sure the response does NOT attempt to answer the user's query or provide any information related to the query, "
            f"and that it clearly and politely states that it can only assist with questions related to '{CHATBOT_THEME}'.\n\n"
            f"User Query: {query}\n\n"
            f"Draft Response to Evaluate: {draft}\n\n"
            f"If the response is a correct and polite refusal, output exactly: PASS\n"
            f"Otherwise, output a detailed explanation of what is wrong with the response."
        )
    else:
        critique_prompt = (
            f"You are a strict quality control evaluator.\n"
            f"Your task is to verify if the draft response is fully grounded in the retrieved documents context.\n"
            f"Make sure there are no hallucinated facts or statements that cannot be verified by the documents.\n\n"
            f"Retrieved Context:\n{docs}\n\n"
            f"User Query: {query}\n\n"
            f"Draft Response to Evaluate: {draft}\n\n"
            f"If the response is fully grounded and correct, output exactly: PASS\n"
            f"Otherwise, output a detailed explanation of what is wrong or hallucinated in the response."
        )
    
    messages = [SystemMessage(content=critique_prompt)]
    response = llm.invoke(messages)
    
    content = response.content.strip()
    if "pass" in content.lower():
        status = "PASS"
        reason = None
    else:
        status = "FAIL"
        reason = content
            
    print(f"Critique result: '{status}'")
    if status == "PASS":
        return {"critique_feedback": "PASS"}
    else:
        print(f"Rejection reason: {reason}")
        return {
            "critique_feedback": reason,
            "attempts": attempts + 1
        }

