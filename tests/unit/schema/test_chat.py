"""Unit tests for chat schema models."""

from datetime import datetime, UTC
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schema.llm.chat import Chat, ChatMessage, ChatRole


class TestChatRole:
    """Tests for ChatRole enum."""

    def test_chat_role_values(self):
        """Test that ChatRole has all expected values."""
        assert ChatRole.USER == "user"
        assert ChatRole.ASSISTANT == "assistant"
        assert ChatRole.SYSTEM == "system"
        assert ChatRole.TOOL == "tool"

    def test_chat_role_members(self):
        """Test that ChatRole has exactly the expected members."""
        expected_roles = {"USER", "ASSISTANT", "SYSTEM", "TOOL"}
        actual_roles = {role.name for role in ChatRole}
        assert actual_roles == expected_roles


class TestChat:
    """Tests for Chat model."""

    def test_chat_creation_with_defaults(self):
        """Test creating a Chat with default values."""
        user_id = uuid4()
        chat = Chat(user_id=user_id)

        assert isinstance(chat.id, UUID)
        assert chat.user_id == user_id
        assert isinstance(chat.created_at, datetime)
        # Verify that created_at is recent (within last minute)
        time_diff = datetime.now(UTC) - chat.created_at
        assert time_diff.total_seconds() < 60

    def test_chat_creation_with_explicit_values(self):
        """Test creating a Chat with explicit values."""
        chat_id = uuid4()
        user_id = uuid4()
        created_at = datetime(2024, 1, 1, 12, 0, 0)

        chat = Chat(
            id=chat_id,
            user_id=user_id,
            created_at=created_at
        )

        assert chat.id == chat_id
        assert chat.user_id == user_id
        assert chat.created_at == created_at

    def test_chat_from_attributes(self):
        """Test that Chat can be created from ORM attributes."""
        # Simulate ORM object with attributes
        class MockORMChat:
            id = uuid4()
            user_id = uuid4()
            created_at = datetime.now(UTC)

        orm_chat = MockORMChat()
        chat = Chat.model_validate(orm_chat, from_attributes=True)

        assert chat.id == orm_chat.id
        assert chat.user_id == orm_chat.user_id
        assert chat.created_at == orm_chat.created_at

    def test_chat_missing_user_id(self):
        """Test that missing user_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Chat()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'user_id' for error in errors)

    def test_chat_invalid_user_id(self):
        """Test that invalid user_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Chat(user_id="not-a-uuid")

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'user_id' for error in errors)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_chat_message_creation_minimal(self):
        """Test creating a ChatMessage with minimal required fields."""
        chat_id = uuid4()
        message = ChatMessage(
            chat_id=chat_id,
            role=ChatRole.USER,
            content="Hello, world!"
        )

        assert isinstance(message.id, UUID)
        assert message.chat_id == chat_id
        assert message.role == ChatRole.USER
        assert message.content == "Hello, world!"
        assert message.tool_calls is None
        assert message.tool_call_id is None
        assert isinstance(message.created_at, datetime)
        assert message.embedding is None

    def test_chat_message_creation_full(self):
        """Test creating a ChatMessage with all fields."""
        message_id = uuid4()
        chat_id = uuid4()
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        tool_calls = {
            "function": "get_weather",
            "arguments": {"location": "San Francisco"}
        }

        message = ChatMessage(
            id=message_id,
            chat_id=chat_id,
            role=ChatRole.ASSISTANT,
            content="I'll check the weather for you.",
            tool_calls=tool_calls,
            tool_call_id="call_123",
            created_at=created_at,
            embedding=0.5
        )

        assert message.id == message_id
        assert message.chat_id == chat_id
        assert message.role == ChatRole.ASSISTANT
        assert message.content == "I'll check the weather for you."
        assert message.tool_calls == tool_calls
        assert message.tool_call_id == "call_123"
        assert message.created_at == created_at
        assert message.embedding == 0.5

    def test_chat_message_all_roles(self):
        """Test that ChatMessage accepts all valid roles."""
        chat_id = uuid4()

        for role in ChatRole:
            message = ChatMessage(
                chat_id=chat_id,
                role=role,
                content=f"Message from {role.value}"
            )
            assert message.role == role

    def test_chat_message_tool_message(self):
        """Test creating a tool message."""
        chat_id = uuid4()
        message = ChatMessage(
            chat_id=chat_id,
            role=ChatRole.TOOL,
            content="Weather data: 72°F, sunny",
            tool_call_id="call_123"
        )

        assert message.role == ChatRole.TOOL
        assert message.tool_call_id == "call_123"

    def test_chat_message_from_attributes(self):
        """Test that ChatMessage can be created from ORM attributes."""
        # Simulate ORM object
        class MockORMMessage:
            id = uuid4()
            chat_id = uuid4()
            role = ChatRole.USER
            content = "Test message"
            tool_calls = None
            tool_call_id = None
            created_at = datetime.now(UTC)
            embedding = None

        orm_message = MockORMMessage()
        message = ChatMessage.model_validate(orm_message, from_attributes=True)

        assert message.id == orm_message.id
        assert message.chat_id == orm_message.chat_id
        assert message.role == orm_message.role
        assert message.content == orm_message.content

    def test_chat_message_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage()

        errors = exc_info.value.errors()
        field_names = {error['loc'][0] for error in errors}
        assert 'chat_id' in field_names
        assert 'role' in field_names
        assert 'content' in field_names

    def test_chat_message_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(
                chat_id=uuid4(),
                role="invalid_role",
                content="Test"
            )

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'role' for error in errors)

    def test_chat_message_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed in tool_calls."""
        chat_id = uuid4()

        # Complex nested structure
        complex_tool_calls = {
            "functions": [
                {"name": "func1", "params": {"a": 1, "b": [1, 2, 3]}},
                {"name": "func2", "params": {"nested": {"deep": True}}}
            ],
            "metadata": {"timestamp": datetime.now(UTC).isoformat()}
        }

        message = ChatMessage(
            chat_id=chat_id,
            role=ChatRole.ASSISTANT,
            content="Using tools",
            tool_calls=complex_tool_calls
        )

        assert message.tool_calls == complex_tool_calls
