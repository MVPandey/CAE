"""Unit tests for app.services.llm_service module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion

from app.schema.llm.message import Message, ToolMessage
from app.schema.llm.tool import ToolCall
from app.services.llm_service import LLMService
from app.utils.exceptions import LLMException


class TestLLMService:
    """Test LLMService class."""

    @pytest.fixture
    def mock_app_settings(self):
        """Mock app settings."""
        with patch("app.services.llm_service.app_settings") as mock_settings:
            mock_settings.LLM_MODEL_NAME = "gpt-4"
            yield mock_settings

    @pytest.fixture
    def llm_service(self, mock_app_settings):
        """Create an LLMService instance."""
        with (
            patch("app.services.llm_service.LLMClient") as mock_client_cls,
            patch("app.services.llm_service.ToolExecutor") as mock_executor_cls,
            patch("app.services.llm_service.tool_registry") as mock_registry,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client_cls.get_retry_decorator.return_value = lambda func: func

            mock_executor = MagicMock()
            mock_executor_cls.return_value = mock_executor

            mock_registry.list_tool_names.return_value = []
            mock_registry.tools = {}

            service = LLMService()
            service.retry_decorator = lambda func: func  # Bypass retry decorator
            return service

    @pytest.fixture
    def mock_completion(self):
        """Create a mock ChatCompletion response."""
        completion = MagicMock(spec=ChatCompletion)
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Test response"
        completion.choices[0].message.tool_calls = None
        completion.choices[0].finish_reason = "stop"
        completion.usage = MagicMock()
        completion.usage.model_dump.return_value = {"total_tokens": 100}
        return completion

    def test_init_default_values(self, mock_app_settings):
        """Test LLMService initialization with default values."""
        with (
            patch("app.services.llm_service.LLMClient") as mock_client_cls,
            patch("app.services.llm_service.ToolExecutor") as mock_executor_cls,
            patch("app.services.llm_service.tool_registry") as mock_registry,
        ):
            mock_registry.list_tool_names.return_value = ["tool1", "tool2"]

            service = LLMService()

            assert service.model_name == "gpt-4"
            mock_client_cls.assert_called_once_with(base_url=None, api_key=None, timeout=None)
            mock_executor_cls.assert_called_once()

    def test_init_custom_values(self):
        """Test LLMService initialization with custom values."""
        with (
            patch("app.services.llm_service.LLMClient") as mock_client_cls,
            patch("app.services.llm_service.ToolExecutor"),
            patch("app.services.llm_service.tool_registry"),
        ):
            service = LLMService(
                base_url="https://api.example.com", api_key="custom-key", model_name="gpt-3.5", timeout=30.0
            )

            assert service.model_name == "gpt-3.5"
            mock_client_cls.assert_called_once_with(
                base_url="https://api.example.com", api_key="custom-key", timeout=30.0
            )

    @pytest.mark.asyncio
    async def test_query_llm_simple_message(self, llm_service, mock_completion):
        """Test query_llm with a simple message."""
        message = Message(role="user", content="Hello")

        with patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)):
            result = await llm_service.query_llm(message)

        assert result == mock_completion.choices[0].message

    @pytest.mark.asyncio
    async def test_query_llm_message_list(self, llm_service, mock_completion):
        """Test query_llm with a list of messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
        ]

        with patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)):
            result = await llm_service.query_llm(messages)

        assert result == mock_completion.choices[0].message

    @pytest.mark.asyncio
    async def test_query_llm_json_response(self, llm_service, mock_completion):
        """Test query_llm with JSON response format."""
        mock_completion.choices[0].message.content = '{"key": "value"}'
        message = Message(role="user", content="Give JSON")

        with patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)):
            result = await llm_service.query_llm(message, json_response=True)

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_query_llm_invalid_json_response(self, llm_service, mock_completion):
        """Test query_llm with invalid JSON response."""
        mock_completion.choices[0].message.content = "Not valid JSON"
        message = Message(role="user", content="Give JSON")

        with (
            patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)),
            patch("app.services.llm_service.clean_json_response") as mock_clean,
        ):
            mock_clean.return_value = {"cleaned": True}

            result = await llm_service.query_llm(message, json_response=True)

        mock_clean.assert_called_once_with("Not valid JSON")
        assert result == {"cleaned": True}

    @pytest.mark.asyncio
    async def test_query_llm_with_tools(self, llm_service, mock_completion):
        """Test query_llm with tools."""
        message = Message(role="user", content="Use tools")

        with (
            patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)),
            patch.object(llm_service, "_prepare_tools") as mock_prepare,
        ):
            mock_prepare.return_value = [{"name": "tool1"}]

            await llm_service.query_llm(message, tools=["tool1"])

        mock_prepare.assert_called_once_with(["tool1"])

    @pytest.mark.asyncio
    async def test_query_llm_with_tool_calls(self, llm_service, mock_completion):
        """Test query_llm when response contains tool calls."""
        mock_tool_call = MagicMock()
        mock_completion.choices[0].message.tool_calls = [mock_tool_call]

        message = Message(role="user", content="Use tool")

        with (
            patch.object(llm_service, "_make_llm_request", AsyncMock(return_value=mock_completion)),
            patch.object(llm_service, "_handle_tool_workflow", AsyncMock(return_value=mock_completion)),
            patch.object(llm_service, "_prepare_tools", return_value=[{"name": "tool1"}]),
        ):
            result = await llm_service.query_llm(message, tools=["tool1"])

        assert result == mock_completion.choices[0].message

    @pytest.mark.asyncio
    async def test_query_llm_parameter_validation(self, llm_service):
        """Test query_llm parameter validation."""
        message = Message(role="user", content="Test")

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            await llm_service.query_llm(message, max_tokens=0)

        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            await llm_service.query_llm(message, temperature=2.5)

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            await llm_service.query_llm(message, top_p=1.5)

    @pytest.mark.asyncio
    async def test_query_llm_exception_handling(self, llm_service):
        """Test query_llm exception handling."""
        message = Message(role="user", content="Test")

        with patch.object(llm_service, "_make_llm_request", AsyncMock(side_effect=Exception("API Error"))):
            with pytest.raises(LLMException, match="Failed to query LLM"):
                await llm_service.query_llm(message)

    def test_normalize_messages_single(self, llm_service):
        """Test _normalize_messages with single message."""
        message = Message(role="user", content="Test")
        result = llm_service._normalize_messages(message)
        assert result == [message]

    def test_normalize_messages_list(self, llm_service):
        """Test _normalize_messages with message list."""
        messages = [Message(role="user", content="Test")]
        result = llm_service._normalize_messages(messages)
        assert result == messages

    def test_prepare_tools_single(self, llm_service):
        """Test _prepare_tools with single tool name."""
        with patch("app.services.llm_service.tool_registry") as mock_registry:
            mock_registry.get_tool_schemas.return_value = [{"name": "tool1"}]

            result = llm_service._prepare_tools("tool1")

            mock_registry.get_tool_schemas.assert_called_once_with(["tool1"])
            assert result == [{"name": "tool1"}]

    def test_prepare_tools_list(self, llm_service):
        """Test _prepare_tools with tool list."""
        with patch("app.services.llm_service.tool_registry") as mock_registry:
            mock_registry.get_tool_schemas.return_value = [{"name": "tool1"}, {"name": "tool2"}]

            result = llm_service._prepare_tools(["tool1", "tool2"])

            mock_registry.get_tool_schemas.assert_called_once_with(["tool1", "tool2"])
            assert result == [{"name": "tool1"}, {"name": "tool2"}]

    def test_tools_property(self, llm_service):
        """Test tools property."""
        with patch("app.services.llm_service.tool_registry") as mock_registry:
            mock_registry.tools = {"tool1": {}}
            assert llm_service.tools == {"tool1": {}}

    @pytest.mark.asyncio
    async def test_make_llm_request(self, llm_service, mock_completion):
        """Test _make_llm_request method."""
        messages = [Message(role="user", content="Test")]
        tools = [{"name": "tool1"}]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        llm_service.client.get_client.return_value = mock_client

        with patch.object(llm_service, "retry_decorator", lambda f: f):
            result = await llm_service._make_llm_request(
                messages=messages, tools=tools, json_response=True, request_id="test-123", max_tokens=500
            )

        assert result == mock_completion
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["max_tokens"] == 500
        assert call_args["tools"] == tools
        assert call_args["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_handle_tool_workflow(self, llm_service, mock_completion):
        """Test _handle_tool_workflow method."""
        initial_completion = MagicMock(spec=ChatCompletion)
        initial_completion.choices = [MagicMock()]
        initial_completion.choices[0].message.tool_calls = [MagicMock()]
        initial_completion.choices[0].message.model_dump.return_value = {"role": "assistant"}

        tool_results = [ToolMessage(role="tool", tool_call_id="123", name="tool1", content="Result")]
        llm_service.tool_executor.execute_tool_calls = AsyncMock(return_value=tool_results)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        llm_service.client.get_client.return_value = mock_client

        messages = [Message(role="user", content="Test")]

        with patch.object(llm_service, "retry_decorator", lambda f: f):
            result = await llm_service._handle_tool_workflow(
                initial_completion=initial_completion,
                messages=messages,
                json_response=False,
                request_id="test-123",
                max_tokens=500,
            )

        assert result == mock_completion
        llm_service.tool_executor.execute_tool_calls.assert_called_once()

    def test_process_response_text(self, llm_service, mock_completion):
        """Test _process_response for text responses."""
        result = llm_service._process_response(mock_completion, False, "test-123")
        assert result == mock_completion.choices[0].message

    def test_process_response_json(self, llm_service, mock_completion):
        """Test _process_response for JSON responses."""
        mock_completion.choices[0].message.content = '{"result": true}'
        result = llm_service._process_response(mock_completion, True, "test-123")
        assert result == {"result": True}

    def test_process_response_none_content(self, llm_service, mock_completion):
        """Test _process_response with None content for JSON."""
        mock_completion.choices[0].message.content = None
        with pytest.raises(LLMException, match="LLM returned None content"):
            llm_service._process_response(mock_completion, True, "test-123")

    def test_process_response_empty_content(self, llm_service, mock_completion):
        """Test _process_response with empty content for JSON."""
        mock_completion.choices[0].message.content = "  "
        with pytest.raises(LLMException, match="LLM returned empty content"):
            llm_service._process_response(mock_completion, True, "test-123")

    @pytest.mark.asyncio
    async def test_handle_tool_calls_backward_compatibility(self, llm_service):
        """Test handle_tool_calls method for backward compatibility."""
        tool_calls = [MagicMock(spec=ToolCall)]
        expected_results = [ToolMessage(role="tool", tool_call_id="123", name="tool1", content="Result")]

        llm_service.tool_executor.execute_tool_calls = AsyncMock(return_value=expected_results)

        result = await llm_service.handle_tool_calls(tool_calls)

        assert result == expected_results
        llm_service.tool_executor.execute_tool_calls.assert_called_once_with(tool_calls)

    @pytest.mark.asyncio
    async def test_query_llm_with_additional_kwargs(self, llm_service, mock_completion):
        """Test query_llm with additional keyword arguments."""
        message = Message(role="user", content="Test")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        llm_service.client.get_client.return_value = mock_client

        with patch.object(llm_service, "retry_decorator", lambda f: f):
            await llm_service.query_llm(
                message, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3
            )

        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["temperature"] == 0.7
        assert call_args["top_p"] == 0.9
        assert call_args["frequency_penalty"] == 0.5
        assert call_args["presence_penalty"] == 0.3
