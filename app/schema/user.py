from datetime import datetime, UTC
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    """Represents a user in the system."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Primary key")
    name: str = Field(..., description="User's name", min_length=1, max_length=255)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UserCreate(BaseModel):
    """Request model for creating a new user."""

    name: str = Field(..., description="User's name", min_length=1, max_length=255)
