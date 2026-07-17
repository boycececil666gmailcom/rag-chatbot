from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

class MessageSchema(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, description="Message content cannot be empty")

class QueryRequest(BaseModel):
    message: str = Field(min_length=1, description="Query message cannot be empty")
    history: List[MessageSchema] = Field(default_factory=list, description="Chat history messages")

class QueryResponse(BaseModel):
    response: str
    tool_calls_executed: List[str] = Field(default_factory=list)

class IngestRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw document text to ingest")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Metadata key-value pairs")

class IngestResponse(BaseModel):
    status: str
    chunk_count: int = Field(ge=0)
