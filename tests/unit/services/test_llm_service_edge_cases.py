"""Edge case tests for LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schema.llm.message import Message
from app.services.llm_service import LLMService


class TestLLMServiceEdgeCases:
    """Test edge cases for LLM service."""

    @pytest.mark.asyncio
    async def test_extract_json_from_response_plain_json(self):
        """Test extracting JSON when response is already plain JSON."""
        service = LLMService(base_url="http://test", api_key="test-key", model_name="test-model")

        response = '{"key": "value", "number": 42}'

        result = await service._extract_json_from_response(response)
        assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    @patch("app.services.llm_service.logger")
    async def test_process_tool_calls_exception(self, mock_logger):
        """Test _process_tool_calls when tool execution fails."""
        service = LLMService(base_url="http://test", api_key="test-key", model_name="test-model")

        service.tool_executor = AsyncMock()
        service.tool_executor.execute_tool_calls = AsyncMock(side_effect=Exception("Tool error"))

        mock_tool_call = MagicMock()
        mock_tool_call.id = MagicMock()
        mock_tool_call.type = "function"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = MagicMock()
        mock_tool_call.function.arguments = MagicMock()

        [Message(role="user", content="test")]

        result = service._process_tool_calls([mock_tool_call], "test-request")

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == mock_tool_call.id
        assert result[0]["type"] == "function"
