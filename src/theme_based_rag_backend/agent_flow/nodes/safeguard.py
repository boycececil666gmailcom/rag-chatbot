from langchain_core.messages import SystemMessage, HumanMessage
from src.theme_based_rag_backend.config import CHATBOT_THEME
from src.theme_based_rag_backend.agent_flow.state import AgentState

def safeguard_node(state: AgentState) -> dict:
    from src.theme_based_rag_backend.agent_flow import llm
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Executing Safeguard refusal\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    refusal_prompt = (
        f"You are a customer service assistant bound to the theme '{CHATBOT_THEME}'.\n"
        f"Politely explain to the user that you are only configured to assist with questions "
        f"related to '{CHATBOT_THEME}', and decline to answer this query."
    )
    messages = [
        SystemMessage(content=refusal_prompt),
        HumanMessage(content=state["message"])
    ]
    response = llm.invoke(messages)
    return {"draft_response": response.content}
