from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ToolMessage(Message):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str
