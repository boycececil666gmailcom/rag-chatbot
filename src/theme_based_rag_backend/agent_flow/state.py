from typing import TypedDict, List, Literal, Optional

class AgentState(TypedDict):
    message: str
    history: List[dict]
    category: Literal["rag", "refuse"]
    retrieved_documents: Optional[str]
    draft_response: str
    critique_feedback: Optional[str]
    attempts: int
