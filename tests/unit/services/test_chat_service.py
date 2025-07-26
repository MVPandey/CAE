"""Unit tests for app.services.chat_service module."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schema.llm.chat import ChatMessage, ChatRole
from app.schema.llm.message import Message as LLMMessage
from app.services.chat_service import ChatService


class TestChatService:
    """Test ChatService class."""

    @pytest.fixture
    def chat_service(self):
        """Create a ChatService instance with mocked dependencies."""
        with patch("app.services.chat_service.LLMService") as mock_llm_service_class:
            mock_llm_service = MagicMock()
            mock_llm_service.tools = {"TestTool": {}}
            mock_llm_service_class.return_value = mock_llm_service
            service = ChatService()
            service.llm_service = mock_llm_service
            return service

    @pytest.fixture
    def mock_uuid(self):
        """Generate a consistent UUID for testing."""
        return uuid4()

    @pytest.fixture
    def mock_chat_message(self, mock_uuid):
        """Create a mock chat message."""
        return ChatMessage(
            chat_id=mock_uuid,
            role=ChatRole.USER,
            content="Test message"
        )

    @pytest.mark.asyncio
    async def test_init(self):
        """Test ChatService initialization."""
        with patch("app.services.chat_service.LLMService") as mock_llm_service_class:
            service = ChatService()
            assert service.llm_service is not None
            mock_llm_service_class.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_session")
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_process_message_new_chat(
        self,
        mock_get_history,
        mock_create_message,
        mock_create_session,
        chat_service,
        mock_uuid
    ):
        """Test processing message with new chat session."""
        mock_session = MagicMock()
        mock_session.id = mock_uuid
        mock_create_session.return_value = mock_session

        mock_history = [
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Hello")
        ]
        mock_get_history.return_value = mock_history

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Hi there!"
        mock_llm_response.tool_calls = None

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)):
            result = await chat_service.process_message(
                chat_id=None,
                user_id=mock_uuid,
                user_message="Hello"
            )

        mock_create_session.assert_called_once_with(mock_uuid)
        assert mock_create_message.call_count == 2  # User message + assistant message
        mock_get_history.assert_called_with(mock_uuid)
        assert result == mock_history

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_process_message_existing_chat(
        self,
        mock_get_history,
        mock_create_message,
        chat_service,
        mock_uuid
    ):
        """Test processing message with existing chat session."""
        mock_history = [
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Hello"),
            ChatMessage(chat_id=mock_uuid, role=ChatRole.ASSISTANT, content="Hi!"),
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="How are you?")
        ]
        mock_get_history.return_value = mock_history

        mock_llm_response = MagicMock()
        mock_llm_response.content = "I'm doing well!"
        mock_llm_response.tool_calls = None

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)):
            result = await chat_service.process_message(
                chat_id=mock_uuid,
                user_id=mock_uuid,
                user_message="How are you?"
            )

        assert mock_create_message.call_count == 2  # User message + assistant message
        mock_get_history.assert_called_with(mock_uuid)
        assert result == mock_history

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_process_message_with_tool_calls(
        self,
        mock_get_history,
        mock_create_message,
        chat_service,
        mock_uuid
    ):
        """Test processing message with tool calls in response."""
        mock_history = [
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Get weather")
        ]
        mock_get_history.return_value = mock_history

        mock_tool_call = MagicMock()
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}'
            }
        }

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Let me check the weather."
        mock_llm_response.tool_calls = [mock_tool_call]

        chat_service.llm_service.tools = {"get_weather": {}}

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)):
            await chat_service.process_message(
                chat_id=mock_uuid,
                user_id=mock_uuid,
                user_message="Get weather"
            )

        assert mock_create_message.call_count == 2

        assistant_msg_call = mock_create_message.call_args_list[1]
        assistant_msg = assistant_msg_call[0][0]
        assert assistant_msg.tool_calls == {"tool_calls": [mock_tool_call.model_dump.return_value]}

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_process_message_empty_response(
        self,
        mock_get_history,
        mock_create_message,
        chat_service,
        mock_uuid
    ):
        """Test processing message with empty LLM response."""
        mock_history = [
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Hello")
        ]
        mock_get_history.return_value = mock_history

        mock_llm_response = MagicMock()
        mock_llm_response.content = None
        mock_llm_response.tool_calls = None

        chat_service.llm_service.tools = {}

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)):
            await chat_service.process_message(
                chat_id=mock_uuid,
                user_id=mock_uuid,
                user_message="Hello"
            )

        assistant_msg_call = mock_create_message.call_args_list[1]
        assistant_msg = assistant_msg_call[0][0]
        assert assistant_msg.content == ""

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_llm_messages_conversion(
        self,
        mock_get_history,
        mock_create_message,
        chat_service,
        mock_uuid
    ):
        """Test proper conversion of chat history to LLM messages."""
        mock_history = [
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Hello"),
            ChatMessage(chat_id=mock_uuid, role=ChatRole.ASSISTANT, content="Hi!"),
            ChatMessage(chat_id=mock_uuid, role=ChatRole.USER, content="Test")
        ]
        mock_get_history.return_value = mock_history

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Response"
        mock_llm_response.tool_calls = None

        chat_service.llm_service.tools = {}

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)) as mock_query:
            await chat_service.process_message(
                chat_id=mock_uuid,
                user_id=mock_uuid,
                user_message="Test"
            )

        llm_messages = mock_query.call_args[0][0]
        assert len(llm_messages) == 3
        assert all(isinstance(msg, LLMMessage) for msg in llm_messages)
        assert llm_messages[0].role == "user"
        assert llm_messages[0].content == "Hello"
        assert llm_messages[1].role == "assistant"
        assert llm_messages[1].content == "Hi!"
        assert llm_messages[2].role == "user"
        assert llm_messages[2].content == "Test"

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_session")
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_process_message_llm_error(
        self,
        mock_get_history,
        mock_create_message,
        mock_create_session,
        chat_service,
        mock_uuid
    ):
        """Test handling of LLM service errors."""
        mock_session = MagicMock()
        mock_session.id = mock_uuid
        mock_create_session.return_value = mock_session
        mock_get_history.return_value = []

        chat_service.llm_service.tools = {}

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(side_effect=Exception("LLM Error"))):
            with pytest.raises(Exception, match="LLM Error"):
                await chat_service.process_message(
                    chat_id=None,
                    user_id=mock_uuid,
                    user_message="Hello"
                )

        assert mock_create_message.call_count == 1

    @pytest.mark.asyncio
    @patch("app.services.chat_service.create_chat_message")
    @patch("app.services.chat_service.get_chat_history")
    async def test_tool_calls_serialization(
        self,
        mock_get_history,
        mock_create_message,
        chat_service,
        mock_uuid
    ):
        """Test proper serialization of multiple tool calls."""
        mock_history = []
        mock_get_history.return_value = mock_history

        tool_calls = []
        for i in range(3):
            mock_tool_call = MagicMock()
            mock_tool_call.model_dump.return_value = {
                "id": f"call_{i}",
                "function": {
                    "name": f"tool_{i}",
                    "arguments": f'{{"param": "{i}"}}'
                }
            }
            tool_calls.append(mock_tool_call)

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Using multiple tools"
        mock_llm_response.tool_calls = tool_calls

        chat_service.llm_service.tools = {"tool_0": {}, "tool_1": {}, "tool_2": {}}

        with patch.object(chat_service.llm_service, "query_llm", AsyncMock(return_value=mock_llm_response)):
            await chat_service.process_message(
                chat_id=mock_uuid,
                user_id=mock_uuid,
                user_message="Do multiple things"
            )

        assistant_msg_call = mock_create_message.call_args_list[1]
        assistant_msg = assistant_msg_call[0][0]
        assert assistant_msg.tool_calls["tool_calls"] == [tc.model_dump.return_value for tc in tool_calls]
        assert len(assistant_msg.tool_calls["tool_calls"]) == 3
