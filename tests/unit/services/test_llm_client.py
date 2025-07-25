"""Unit tests for app.services.llm.client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
import tenacity
from openai import AsyncOpenAI
from tenacity import RetryCallState

from app.services.llm.client import LLMClient
from app.utils.constants import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)


class TestLLMClient:
    """Test LLMClient class."""

    @pytest.fixture
    def mock_app_settings(self):
        """Mock app settings."""
        with patch("app.services.llm.client.app_settings") as mock_settings:
            mock_settings.LLM_API_BASE_URL = "https://api.openai.com"
            mock_settings.LLM_API_KEY = "test-key"
            mock_settings.LLM_TIMEOUT_SECONDS = "60"
            yield mock_settings

    def test_init_default_values(self, mock_app_settings):
        """Test LLMClient initialization with default values."""
        client = LLMClient()

        assert client.base_url == "https://api.openai.com"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0
        assert client.max_retries == DEFAULT_MAX_RETRIES

    def test_init_custom_values(self):
        """Test LLMClient initialization with custom values."""
        client = LLMClient(
            base_url="https://custom.api.com",
            api_key="custom-key",
            timeout=30.0,
            max_retries=5
        )

        assert client.base_url == "https://custom.api.com"
        assert client.api_key == "custom-key"
        assert client.timeout == 30.0
        assert client.max_retries == 5

    def test_init_none_timeout(self, mock_app_settings):
        """Test LLMClient initialization when timeout is None in settings."""
        mock_app_settings.LLM_TIMEOUT_SECONDS = None

        client = LLMClient()
        assert client.timeout == float(DEFAULT_LLM_TIMEOUT)

    def test_get_client(self, mock_app_settings):
        """Test get_client returns configured AsyncOpenAI instance."""
        client = LLMClient()

        with patch("app.services.llm.client.AsyncOpenAI") as mock_openai_cls:
            mock_openai_instance = MagicMock(spec=AsyncOpenAI)
            mock_openai_cls.return_value = mock_openai_instance

            result = client.get_client()

            mock_openai_cls.assert_called_once_with(
                base_url="https://api.openai.com",
                api_key="test-key",
                timeout=60.0,
                max_retries=DEFAULT_MAX_RETRIES
            )
            assert result == mock_openai_instance

    @patch("app.services.llm.client.logger")
    def test_log_retry_attempt_first_attempt(self, mock_logger):
        """Test _log_retry_attempt on first attempt (should not log)."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 1

        LLMClient._log_retry_attempt(retry_state)

        mock_logger.warning.assert_not_called()

    @patch("app.services.llm.client.logger")
    def test_log_retry_attempt_subsequent_attempts(self, mock_logger):
        """Test _log_retry_attempt on subsequent attempts."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 2
        retry_state.next_action = MagicMock(sleep=5)
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = Exception("Test error")

        LLMClient._log_retry_attempt(retry_state)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Retrying LLM request" in call_args[0]
        assert call_args[1]["extra"]["attempt"] == 2
        assert call_args[1]["extra"]["wait_time"] == 5
        assert "Test error" in call_args[1]["extra"]["exception"]

    @patch("app.services.llm.client.logger")
    def test_log_retry_attempt_no_next_action(self, mock_logger):
        """Test _log_retry_attempt when next_action is None."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 2
        retry_state.next_action = None
        retry_state.outcome = None

        LLMClient._log_retry_attempt(retry_state)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[1]["extra"]["wait_time"] is None
        assert call_args[1]["extra"]["exception"] is None

    def test_get_retry_decorator(self):
        """Test get_retry_decorator returns proper retry configuration."""
        decorator = LLMClient.get_retry_decorator()

        # Verify it's a retry decorator by checking it's a callable that wraps functions
        assert callable(decorator)

        # Test that it properly decorates a function
        @decorator
        def test_func():
            pass

        # The decorated function should have retry attributes from tenacity
        assert hasattr(test_func, "retry")
        assert hasattr(test_func, "retry_with")

    @pytest.mark.asyncio
    @patch("app.services.llm.client.logger")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_get_retry_decorator_with_rate_limit_error(self, mock_sleep, mock_logger):
        """Test retry decorator handles RateLimitError."""
        decorator = LLMClient.get_retry_decorator()

        # Create a mock function that raises RateLimitError
        @decorator
        async def test_function():
            # Create a mock response object
            mock_response = MagicMock()
            mock_response.status_code = 429
            raise openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)

        # The decorator should retry on RateLimitError
        with pytest.raises(tenacity.RetryError) as exc_info:
            # This will retry RETRY_MAX_ATTEMPTS times before re-raising
            await test_function()

        # The original exception should be wrapped in RetryError
        assert isinstance(exc_info.value.last_attempt.exception(), openai.RateLimitError)
        # Should have logged retry attempts (only logs for attempt > 1)
        # Simulate a fixed number of retries and verify the expected log calls
        expected_retry_attempts = 4  # RETRY_MAX_ATTEMPTS - 1
        assert mock_logger.warning.call_count == expected_retry_attempts

    @pytest.mark.asyncio
    @patch("app.services.llm.client.logger")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_get_retry_decorator_with_timeout_error(self, mock_sleep, mock_logger):
        """Test retry decorator handles APITimeoutError."""
        decorator = LLMClient.get_retry_decorator()

        @decorator
        async def test_function():
            # APITimeoutError requires a request parameter
            mock_request = MagicMock()
            raise openai.APITimeoutError(request=mock_request)

        with pytest.raises(tenacity.RetryError) as exc_info:
            await test_function()

        # The original exception should be wrapped in RetryError
        assert isinstance(exc_info.value.last_attempt.exception(), openai.APITimeoutError)
        expected_retry_attempts = 4  # RETRY_MAX_ATTEMPTS - 1
        assert mock_logger.warning.call_count == expected_retry_attempts

    @pytest.mark.asyncio
    @patch("app.services.llm.client.logger")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_get_retry_decorator_with_connection_error(self, mock_sleep, mock_logger):
        """Test retry decorator handles APIConnectionError."""
        decorator = LLMClient.get_retry_decorator()

        @decorator
        async def test_function():
            # APIConnectionError requires a request parameter
            mock_request = MagicMock()
            raise openai.APIConnectionError(request=mock_request)

        with pytest.raises(tenacity.RetryError) as exc_info:
            await test_function()

        # The original exception should be wrapped in RetryError
        assert isinstance(exc_info.value.last_attempt.exception(), openai.APIConnectionError)
        expected_retry_attempts = 4  # RETRY_MAX_ATTEMPTS - 1
        assert mock_logger.warning.call_count == expected_retry_attempts

    @pytest.mark.asyncio
    @patch("app.services.llm.client.logger")
    async def test_get_retry_decorator_non_retryable_error(self, mock_logger):
        """Test retry decorator does not retry non-retryable errors."""
        decorator = LLMClient.get_retry_decorator()

        @decorator
        async def test_function():
            raise ValueError("Non-retryable error")

        # Should raise immediately without retrying
        with pytest.raises(ValueError):
            await test_function()

        # Should not have logged any retry attempts
        mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.llm.client.logger")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_get_retry_decorator_successful_after_retry(self, mock_sleep, mock_logger):
        """Test retry decorator succeeds after retries."""
        decorator = LLMClient.get_retry_decorator()

        call_count = 0

        @decorator
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Create a mock response object
                mock_response = MagicMock()
                mock_response.status_code = 429
                raise openai.RateLimitError("Rate limit", response=mock_response, body=None)
            return "success"

        result = await test_function()

        assert result == "success"
        assert call_count == 3
        # _log_retry_attempt logs for attempt_number > 1, which happens before attempts 2 and 3
        # But since attempt 3 succeeds, we may only see 1 log (for attempt 2)
        assert mock_logger.warning.call_count >= 1

    @patch("app.services.llm.client.logger")
    def test_logging_on_init(self, mock_logger, mock_app_settings):
        """Test that initialization logs debug information."""
        LLMClient(
            base_url="https://test.com",
            api_key="test-key",
            timeout=30.0,
            max_retries=3
        )

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "Initialized LLM client" in call_args[0][0]
        assert call_args[1]["extra"]["base_url"] == "https://test.com"
        assert call_args[1]["extra"]["timeout"] == 30.0
        assert call_args[1]["extra"]["max_retries"] == 3
