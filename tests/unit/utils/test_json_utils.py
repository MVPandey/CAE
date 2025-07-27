"""Unit tests for app.utils.json_utils module."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from app.utils.exceptions import LLMException
from app.utils.json_utils import clean_json_response, safe_json_dumps


class TestCleanJsonResponse:
    """Test clean_json_response function."""

    def test_clean_json_response_valid_json(self):
        """Test parsing valid JSON."""
        response = '{"key": "value", "number": 42}'
        result = clean_json_response(response)
        assert result == {"key": "value", "number": 42}

    def test_clean_json_response_empty_string(self):
        """Test empty response raises exception."""
        with pytest.raises(LLMException) as exc_info:
            clean_json_response("")
        assert "Empty response received" in str(exc_info.value)

    def test_clean_json_response_none(self):
        """Test None response raises exception."""
        with pytest.raises(LLMException) as exc_info:
            clean_json_response(None)
        assert "Empty response received" in str(exc_info.value)

    def test_clean_json_response_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        response = """Here's the JSON:
        ```json
        {
            "name": "test",
            "value": 123
        }
        ```
        """
        result = clean_json_response(response)
        assert result == {"name": "test", "value": 123}

    def test_clean_json_response_markdown_block_no_language(self):
        """Test extracting JSON from markdown block without language specification."""
        response = """
        ```
        {"items": ["a", "b", "c"]}
        ```
        """
        result = clean_json_response(response)
        assert result == {"items": ["a", "b", "c"]}

    def test_clean_json_response_trailing_commas(self):
        """Test handling JSON with trailing commas."""
        response = '{"key1": "value1", "key2": "value2",}'
        result = clean_json_response(response)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_clean_json_response_array_trailing_comma(self):
        """Test handling JSON array with trailing comma."""
        response = '["item1", "item2", "item3",]'
        result = clean_json_response(response)
        assert result == ["item1", "item2", "item3"]

    def test_clean_json_response_mixed_content(self):
        """Test extracting JSON from mixed content."""
        response = """The result is as follows:
        {"status": "success", "data": [1, 2, 3]}
        That's the response."""
        result = clean_json_response(response)
        assert result == {"status": "success", "data": [1, 2, 3]}

    def test_clean_json_response_nested_json(self):
        """Test parsing nested JSON structures."""
        response = '{"outer": {"inner": {"value": 42}, "array": [1, 2, 3]}}'
        result = clean_json_response(response)
        assert result == {"outer": {"inner": {"value": 42}, "array": [1, 2, 3]}}

    def test_clean_json_response_invalid_json(self):
        """Test completely invalid JSON raises exception."""
        response = "This is not JSON at all"
        with pytest.raises(LLMException) as exc_info:
            clean_json_response(response)
        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)
        assert response in str(exc_info.value.details["response"])

    def test_clean_json_response_malformed_json(self):
        """Test malformed JSON raises exception."""
        response = '{"key": "value" "missing_comma": "value"}'
        with pytest.raises(LLMException) as exc_info:
            clean_json_response(response)
        assert "Failed to parse JSON response after all attempts" in str(exc_info.value)

    @patch("app.utils.json_utils.logger")
    def test_clean_json_response_logging(self, mock_logger):
        """Test that errors are logged appropriately."""
        response = "```json\n{invalid json}\n```"
        with pytest.raises(LLMException):
            clean_json_response(response)

        assert mock_logger.error.called
        error_calls = mock_logger.error.call_args_list
        assert any("Failed to parse JSON from markdown block" in str(call) for call in error_calls)

    def test_clean_json_response_large_response(self):
        """Test handling of large responses in error details."""
        large_response = "x" * 2000
        with pytest.raises(LLMException) as exc_info:
            clean_json_response(large_response)

        assert len(exc_info.value.details["response"]) <= 1003


class TestSafeJsonDumps:
    """Test safe_json_dumps function."""

    def test_safe_json_dumps_basic_types(self):
        """Test serializing basic Python types."""
        assert safe_json_dumps({"key": "value"}) == '{"key": "value"}'
        assert safe_json_dumps([1, 2, 3]) == "[1, 2, 3]"
        assert safe_json_dumps("string") == '"string"'
        assert safe_json_dumps(42) == "42"
        assert safe_json_dumps(3.14) == "3.14"
        assert safe_json_dumps(True) == "true"
        assert safe_json_dumps(None) == "null"

    def test_safe_json_dumps_with_kwargs(self):
        """Test passing additional kwargs to json.dumps."""
        data = {"a": 1, "b": 2}
        result = safe_json_dumps(data, indent=2, sort_keys=True)
        expected = '{\n  "a": 1,\n  "b": 2\n}'
        assert result == expected

    def test_safe_json_dumps_non_serializable(self):
        """Test handling non-serializable objects."""
        obj = datetime.now()
        result = safe_json_dumps(obj)

        parsed = json.loads(result)
        assert isinstance(parsed, str)
        assert str(obj) in parsed

    @patch("app.utils.json_utils.logger")
    def test_safe_json_dumps_error_logging(self, mock_logger):
        """Test that serialization errors are logged."""

        class NonSerializable:
            def __str__(self):
                return "NonSerializable object"

        obj = NonSerializable()
        result = safe_json_dumps(obj)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Failed to serialize object to JSON" in call_args[0]
        assert call_args[1]["extra"]["object_type"] == "NonSerializable"

        assert result == '"NonSerializable object"'

    def test_safe_json_dumps_complex_nested(self):
        """Test serializing complex nested structures."""
        data = {
            "users": [{"id": 1, "name": "Alice", "active": True}, {"id": 2, "name": "Bob", "active": False}],
            "metadata": {"version": "1.0", "count": 2},
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_safe_json_dumps_with_custom_encoder(self):
        """Test that custom encoder arguments work."""
        data = {"value": float("inf")}

        result = safe_json_dumps(data, allow_nan=False)

        parsed = json.loads(result)
        assert isinstance(parsed, str)
