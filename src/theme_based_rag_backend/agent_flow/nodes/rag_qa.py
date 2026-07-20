from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.theme_based_rag_backend.config import CHATBOT_THEME
from src.theme_based_rag_backend.agent_flow.state import AgentState

def rag_qa_node(state: AgentState) -> dict:
    from src.theme_based_rag_backend.agent_flow import llm, retrieve_local_documents
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
    content = response.content
    if isinstance(content, list):
        content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    return {
        "draft_response": content,
        "retrieved_documents": retrieved_docs
    }
