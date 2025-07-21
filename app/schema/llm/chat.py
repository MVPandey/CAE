from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Chat(BaseModel):
    """Represents a chat session."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Primary key")
    user_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    """Represents a single message in a chat."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4, description="Primary key")
    chat_id: UUID = Field(description="Foreign key to Chat.id")
    role: ChatRole
    content: str
    tool_calls: Optional[dict[str, Any]] = Field(default=None, description="JSONB field")
    tool_call_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[float] = Field(default=None, description="Vector with 4096 dimensions")
