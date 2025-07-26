"""Unit tests for message schema models."""

import pytest
from pydantic import ValidationError

from app.schema.llm.message import Message, ToolMessage


class TestMessage:
    """Tests for Message model."""

    def test_message_creation_user(self):
        """Test creating a user message."""
        message = Message(role="user", content="Hello, assistant!")
        assert message.role == "user"
        assert message.content == "Hello, assistant!"

    def test_message_creation_assistant(self):
        """Test creating an assistant message."""
        message = Message(role="assistant", content="Hello! How can I help you?")
        assert message.role == "assistant"
        assert message.content == "Hello! How can I help you?"

    def test_message_creation_system(self):
        """Test creating a system message."""
        message = Message(role="system", content="You are a helpful assistant.")
        assert message.role == "system"
        assert message.content == "You are a helpful assistant."

    def test_message_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="invalid", content="Test")

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'role' for error in errors)
        error_str = str(exc_info.value)
        assert "user" in error_str
        assert "assistant" in error_str
        assert "system" in error_str

    def test_message_missing_fields(self):
        """Test that missing fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Message()

        errors = exc_info.value.errors()
        field_names = {error['loc'][0] for error in errors}
        assert 'role' in field_names
        assert 'content' in field_names

    def test_message_empty_content(self):
        """Test that empty content is allowed."""
        message = Message(role="user", content="")
        assert message.content == ""

    def test_message_multiline_content(self):
        """Test message with multiline content."""
        content = """This is a
        multiline
        message content"""
        message = Message(role="user", content=content)
        assert message.content == content


class TestToolMessage:
    """Tests for ToolMessage model."""

    def test_tool_message_creation(self):
        """Test creating a tool message with all fields."""
        tool_message = ToolMessage(
            role="tool",
            tool_call_id="call_abc123",
            name="get_weather",
            content="Temperature: 72°F, Conditions: Sunny"
        )

        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == "call_abc123"
        assert tool_message.name == "get_weather"
        assert tool_message.content == "Temperature: 72°F, Conditions: Sunny"

    def test_tool_message_inherits_from_message(self):
        """Test that ToolMessage inherits from Message."""
        assert issubclass(ToolMessage, Message)

    def test_tool_message_role_must_be_tool(self):
        """Test that role must be 'tool' for ToolMessage."""
        tool_message = ToolMessage(
            role="tool",
            tool_call_id="123",
            name="test",
            content="content"
        )
        assert tool_message.role == "tool"

        with pytest.raises(ValidationError) as exc_info:
            ToolMessage(
                role="user",  # Invalid role for ToolMessage
                tool_call_id="123",
                name="test",
                content="content"
            )

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'role' for error in errors)

    def test_tool_message_missing_fields(self):
        """Test that missing fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ToolMessage(role="tool")

        errors = exc_info.value.errors()
        field_names = {error['loc'][0] for error in errors}
        assert 'tool_call_id' in field_names
        assert 'name' in field_names
        assert 'content' in field_names

    def test_tool_message_empty_values(self):
        """Test that empty values are allowed for string fields."""
        tool_message = ToolMessage(
            role="tool",
            tool_call_id="",
            name="",
            content=""
        )

        assert tool_message.tool_call_id == ""
        assert tool_message.name == ""
        assert tool_message.content == ""

    def test_tool_message_json_content(self):
        """Test tool message with JSON-like content."""
        json_content = '{"temperature": 72, "unit": "fahrenheit", "conditions": "sunny"}'
        tool_message = ToolMessage(
            role="tool",
            tool_call_id="weather_call_123",
            name="get_current_weather",
            content=json_content
        )

        assert tool_message.content == json_content

    def test_tool_message_error_content(self):
        """Test tool message with error content."""
        error_content = "Error: Unable to fetch weather data. API rate limit exceeded."
        tool_message = ToolMessage(
            role="tool",
            tool_call_id="failed_call_456",
            name="get_current_weather",
            content=error_content
        )

        assert tool_message.content == error_content
