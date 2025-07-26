"""Unit tests for app.services.llm.tool_executor module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schema.llm.message import ToolMessage
from app.schema.llm.tool import ToolCall
from app.services.llm.tool_executor import ToolExecutor
from app.utils.constants import LOG_CONTENT_PREVIEW_LENGTH


class TestToolExecutor:
    """Test ToolExecutor class."""

    @pytest.fixture
    def tool_executor(self):
        """Create a ToolExecutor instance."""
        with patch("app.services.llm.tool_executor.tool_registry") as mock_registry:
            executor = ToolExecutor()
            executor.registry = mock_registry
            return executor

    @pytest.fixture
    def mock_tool_call(self):
        """Create a mock ToolCall."""
        call = MagicMock(spec=ToolCall)
        call.id = "call_123"
        call.function = MagicMock()
        call.function.name = "test_tool"
        call.function.arguments = '{"param": "value"}'
        return call

    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_list(self, tool_executor):
        """Test execute_tool_calls with empty list."""
        result = await tool_executor.execute_tool_calls([])
        assert result == []

    @pytest.mark.asyncio
    @patch("app.services.llm.tool_executor.logger")
    async def test_execute_tool_calls_single_success(self, mock_logger, tool_executor, mock_tool_call):
        """Test execute_tool_calls with single successful call."""
        async def mock_tool_function(param):
            return {"result": param}

        tool_executor.registry.get_tool_function.return_value = mock_tool_function

        results = await tool_executor.execute_tool_calls([mock_tool_call])

        assert len(results) == 1
        assert isinstance(results[0], ToolMessage)
        assert results[0].role == "tool"
        assert results[0].tool_call_id == "call_123"
        assert results[0].name == "test_tool"
        assert json.loads(results[0].content) == {"result": "value"}

        info_calls = mock_logger.info.call_args_list
        assert any("Starting tool execution batch" in str(call) for call in info_calls)
        assert any("Tool execution batch completed" in str(call) for call in info_calls)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple_success(self, tool_executor):
        """Test execute_tool_calls with multiple successful calls."""
        tool_calls = []
        for i in range(3):
            call = MagicMock(spec=ToolCall)
            call.id = f"call_{i}"
            call.function = MagicMock()
            call.function.name = f"tool_{i}"
            call.function.arguments = f'{{"param": "{i}"}}'
            tool_calls.append(call)

        async def mock_tool_function(**kwargs):
            return {"result": kwargs.get("param")}

        tool_executor.registry.get_tool_function.return_value = mock_tool_function

        results = await tool_executor.execute_tool_calls(tool_calls)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.tool_call_id == f"call_{i}"
            assert result.name == f"tool_{i}"
            assert json.loads(result.content) == {"result": str(i)}

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_error(self, tool_executor, mock_tool_call):
        """Test execute_tool_calls with tool execution error."""
        async def failing_tool_function(**kwargs):
            raise RuntimeError("Tool execution failed")

        tool_executor.registry.get_tool_function.return_value = failing_tool_function

        results = await tool_executor.execute_tool_calls([mock_tool_call])

        assert len(results) == 1
        assert results[0].role == "tool"
        assert results[0].tool_call_id == "call_123"

        error_content = json.loads(results[0].content)
        assert error_content["error"] == "Tool execution failed"
        assert error_content["error_type"] == "RuntimeError"
        assert error_content["tool_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_custom_execution_id(self, tool_executor, mock_tool_call):
        """Test execute_tool_calls with custom execution ID."""
        async def mock_tool_function(**kwargs):
            return "result"

        tool_executor.registry.get_tool_function.return_value = mock_tool_function

        results = await tool_executor.execute_tool_calls([mock_tool_call], execution_id="custom-id")

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_execute_single_tool_invalid_json(self, tool_executor):
        """Test _execute_single_tool with invalid JSON arguments."""
        call = MagicMock(spec=ToolCall)
        call.id = "call_123"
        call.function = MagicMock()
        call.function.name = "test_tool"
        call.function.arguments = "invalid json"

        with pytest.raises(ValueError, match="Invalid JSON in tool arguments"):
            await tool_executor._execute_single_tool(call, "exec-123", 0)

    @pytest.mark.asyncio
    async def test_execute_single_tool_non_dict_args(self, tool_executor):
        """Test _execute_single_tool with non-dictionary arguments."""
        call = MagicMock(spec=ToolCall)
        call.id = "call_123"
        call.function = MagicMock()
        call.function.name = "test_tool"
        call.function.arguments = '["not", "a", "dict"]'

        with pytest.raises(ValueError, match="Tool arguments must be a dictionary"):
            await tool_executor._execute_single_tool(call, "exec-123", 0)

    @pytest.mark.asyncio
    async def test_execute_single_tool_unknown_tool(self, tool_executor, mock_tool_call):
        """Test _execute_single_tool with unknown tool."""
        tool_executor.registry.get_tool_function.side_effect = ValueError("Tool not found")
        tool_executor.registry.list_tool_names.return_value = ["other_tool"]

        with pytest.raises(ValueError, match="Tool not found"):
            await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)

    @pytest.mark.asyncio
    async def test_execute_single_tool_various_result_types(self, tool_executor, mock_tool_call):
        """Test _execute_single_tool with various result types."""
        result_with_json = MagicMock()
        result_with_json.json.return_value = '{"type": "json_method"}'

        async def tool_json(**kwargs):
            return result_with_json

        tool_executor.registry.get_tool_function.return_value = tool_json
        result = await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)
        assert result.content == '{"type": "json_method"}'

        result_with_dump = MagicMock()
        result_with_dump.model_dump.return_value = {"type": "model_dump"}
        del result_with_dump.json  # Ensure json method doesn't exist

        async def tool_dump(**kwargs):
            return result_with_dump

        tool_executor.registry.get_tool_function.return_value = tool_dump
        result = await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)
        assert json.loads(result.content) == {"type": "model_dump"}

        async def tool_dict(**kwargs):
            return {"type": "dict"}

        tool_executor.registry.get_tool_function.return_value = tool_dict
        result = await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)
        assert json.loads(result.content) == {"type": "dict"}

        async def tool_string(**kwargs):
            return "simple string"

        tool_executor.registry.get_tool_function.return_value = tool_string
        result = await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)
        assert result.content == '"simple string"'

        class CustomObject:
            def __str__(self):
                return "custom object string"

        async def tool_custom(**kwargs):
            return CustomObject()

        tool_executor.registry.get_tool_function.return_value = tool_custom
        result = await tool_executor._execute_single_tool(mock_tool_call, "exec-123", 0)
        assert result.content == '"custom object string"'

    def test_serialize_result_various_types(self, tool_executor):
        """Test _serialize_result with various input types."""
        obj_with_json = MagicMock()
        obj_with_json.json.return_value = '{"json": true}'
        assert tool_executor._serialize_result(obj_with_json) == '{"json": true}'

        obj_with_dump = MagicMock(spec=["model_dump"])
        obj_with_dump.model_dump.return_value = {"dump": True}
        assert tool_executor._serialize_result(obj_with_dump) == '{"dump": true}'

        assert tool_executor._serialize_result({"dict": True}) == '{"dict": true}'
        assert tool_executor._serialize_result([1, 2, 3]) == '[1, 2, 3]'
        assert tool_executor._serialize_result("string") == '"string"'
        assert tool_executor._serialize_result(123) == '123'
        assert tool_executor._serialize_result(123.45) == '123.45'
        assert tool_executor._serialize_result(True) == 'true'
        assert tool_executor._serialize_result(None) == 'null'

        class Custom:
            def __str__(self):
                return "custom_str"

        assert tool_executor._serialize_result(Custom()) == '"custom_str"'

    def test_create_error_message(self, tool_executor, mock_tool_call):
        """Test _create_error_message."""
        error = RuntimeError("Test error")

        result = tool_executor._create_error_message(mock_tool_call, error, "exec-123", 0)

        assert isinstance(result, ToolMessage)
        assert result.role == "tool"
        assert result.tool_call_id == "call_123"
        assert result.name == "test_tool"

        content = json.loads(result.content)
        assert content["error"] == "Test error"
        assert content["error_type"] == "RuntimeError"
        assert content["tool_name"] == "test_tool"

    def test_create_error_message_missing_attributes(self, tool_executor):
        """Test _create_error_message with missing attributes."""
        broken_call = MagicMock()
        del broken_call.id  # Remove id attribute
        del broken_call.function  # Remove function attribute

        error = Exception("Error")

        result = tool_executor._create_error_message(broken_call, error, "exec-123", 0)

        assert result.tool_call_id == "unknown"
        assert result.name == "unknown"

    def test_preview_content_short(self, tool_executor):
        """Test _preview_content with short content."""
        content = "Short content"
        preview = tool_executor._preview_content(content)
        assert preview == "Short content"

    def test_preview_content_long(self, tool_executor):
        """Test _preview_content with long content."""
        content = "x" * (LOG_CONTENT_PREVIEW_LENGTH + 50)
        preview = tool_executor._preview_content(content)
        assert len(preview) == LOG_CONTENT_PREVIEW_LENGTH + 3  # +3 for "..."
        assert preview.endswith("...")
        assert preview.startswith("x" * LOG_CONTENT_PREVIEW_LENGTH)

    def test_preview_content_custom_length(self, tool_executor):
        """Test _preview_content with custom max length."""
        content = "This is a longer content that should be truncated"
        preview = tool_executor._preview_content(content, max_length=10)
        assert preview == "This is a ..."
        assert len(preview) == 13  # 10 + 3 for "..."

    @pytest.mark.asyncio
    @patch("app.services.llm.tool_executor.logger")
    async def test_logging_details(self, mock_logger, tool_executor, mock_tool_call):
        """Test detailed logging throughout execution."""
        async def mock_tool_function(**kwargs):
            return {"result": "success"}

        tool_executor.registry.get_tool_function.return_value = mock_tool_function

        await tool_executor.execute_tool_calls([mock_tool_call], execution_id="test-exec")

        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        assert any("Executing individual tool call" in call for call in debug_calls)
        assert any("Executing tool function" in call for call in debug_calls)
        assert any("Tool function executed successfully" in call for call in debug_calls)

        assert any("Starting tool execution batch" in call for call in info_calls)
        assert any("Tool call executed successfully" in call for call in info_calls)
        assert any("Tool execution batch completed" in call for call in info_calls)
