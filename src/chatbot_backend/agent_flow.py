import json
import logging
from typing import TypedDict, List, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.chatbot_backend.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, CHATBOT_THEME
from src.chatbot_backend.tools import retrieve_local_documents
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# Initialize LLM
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=GEMINI_TEMPERATURE
)

# 1. State Definition
class AgentState(TypedDict):
    message: str
    history: List[dict]
    category: Literal["rag", "refuse"]
    retrieved_documents: Optional[str]
    draft_response: str
    critique_feedback: Optional[str]
    attempts: int

# 2. Classifier Node
def classifier_node(state: AgentState) -> dict:
    query = state["message"]
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Classifying user query theme\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    messages = [
        SystemMessage(content=(
            f"You are a triage classifier for an AI customer service agent.\n"
            f"Your job is to determine if the query is related to the theme: '{CHATBOT_THEME}'.\n"
            f"You must respond with exactly one of the following JSON formats:\n"
            f'{{"category": "rag"}}\n'
            f'or\n'
            f'{{"category": "refuse"}}\n'
            f"Return ONLY valid JSON."
        )),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:-1]
            content = "\n".join(lines).strip()
        data = json.loads(content)
        category = data.get("category", "refuse")
        if category not in ["rag", "refuse"]:
            category = "refuse"
    except Exception as e:
        logger.warning(f"Failed to parse category response, falling back to substring check: {e}")
        if "rag" in response.content.lower():
            category = "rag"
        else:
            category = "refuse"
            
    print(f"-> Categorized as: '{category}'")
    return {"category": category}

# 3. RAG QA Node
def rag_qa_node(state: AgentState) -> dict:
    query = state["message"]
    history = state.get("history", [])
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Executing RAG QA retrieval & synthesis\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    # Retrieve local documents if not already present
    retrieved_docs = state.get("retrieved_documents")
    if not retrieved_docs:
        print("Invoking retrieve_local_documents tool...")
        retrieved_docs = retrieve_local_documents.invoke(query)
    
    system_prompt = (
        f"You are a customer service assistant. Your primary theme is: {CHATBOT_THEME}.\n"
        f"Answer the user's question using ONLY the provided retrieved document context below.\n\n"
        f"Retrieved Document Context:\n{retrieved_docs}\n\n"
        f"CRITICAL RULES:\n"
        f"1. Your answer must be strictly grounded in the retrieved document context.\n"
        f"2. If the context does not contain the answer, politely state that you cannot find "
        f"the information in the local documentation.\n"
        f"3. Do not make up facts or use pre-trained general knowledge."
    )
    
    messages = [SystemMessage(content=system_prompt)]
    
    # Hydrate history
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    # Include critique feedback if looping back
    feedback = state.get("critique_feedback")
    prev_draft = state.get("draft_response")
    if feedback and prev_draft:
        print(f"Refinement attempt: applying critique feedback: {feedback}")
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=prev_draft))
        refine_msg = (
            f"CRITIQUE FEEDBACK: Your previous draft answer was rejected because: {feedback}\n"
            f"Please revise your answer to address this feedback. Make sure the response is "
            f"fully grounded in the retrieved document context."
        )
        messages.append(SystemMessage(content=refine_msg))
    else:
        messages.append(HumanMessage(content=query))
        
    response = llm.invoke(messages)
    return {
        "draft_response": response.content,
        "retrieved_documents": retrieved_docs
    }

# 4. Safeguard Node
def safeguard_node(state: AgentState) -> dict:
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

# 5. Critique Node
def critique_node(state: AgentState) -> dict:
    category = state.get("category")
    if category == "refuse":
        return {"critique_feedback": "PASS"}
        
    draft = state.get("draft_response")
    docs = state.get("retrieved_documents")
    query = state["message"]
    attempts = state.get("attempts", 0)
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Critique Node validating response groundedness\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    critique_prompt = (
        f"You are a strict quality control evaluator.\n"
        f"Your task is to verify if the draft response is fully grounded in the retrieved documents context.\n"
        f"Make sure there are no hallucinated facts or statements that cannot be verified by the documents.\n\n"
        f"Retrieved Context:\n{docs}\n\n"
        f"User Query: {query}\n\n"
        f"Draft Response to Evaluate: {draft}\n\n"
        f"You must respond in exactly one of the following JSON formats:\n"
        f'{{"status": "PASS"}}\n'
        f'or\n'
        f'{{"status": "FAIL", "reason": "Detailed explanation of why it is not grounded or what facts are missing/hallucinated"}}\n'
        f"Return ONLY valid JSON."
    )
    
    messages = [SystemMessage(content=critique_prompt)]
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:-1]
            content = "\n".join(lines).strip()
        data = json.loads(content)
        status = data.get("status", "FAIL")
        reason = data.get("reason", "Hallucination or lack of grounding detected.")
    except Exception as e:
        logger.warning(f"Failed to parse critique response, falling back to substring check: {e}")
        if "pass" in response.content.lower():
            status = "PASS"
            reason = None
        else:
            status = "FAIL"
            reason = "Failed to parse critique evaluation."
            
    print(f"Critique result: '{status}'")
    if status == "PASS":
        return {"critique_feedback": "PASS"}
    else:
        print(f"Rejection reason: {reason}")
        return {
            "critique_feedback": reason,
            "attempts": attempts + 1
        }

# 6. Routing and Conditional Logic
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

# 7. Workflow Graph Setup
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
