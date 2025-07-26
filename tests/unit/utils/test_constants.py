"""Unit tests for app.utils.constants module."""

from app.utils import constants


class TestConstants:
    """Test constants module values."""

    def test_default_max_tokens(self):
        """Test DEFAULT_MAX_TOKENS constant."""
        assert constants.DEFAULT_MAX_TOKENS == 250
        assert isinstance(constants.DEFAULT_MAX_TOKENS, int)
        assert constants.DEFAULT_MAX_TOKENS > 0

    def test_default_llm_timeout(self):
        """Test DEFAULT_LLM_TIMEOUT constant."""
        assert constants.DEFAULT_LLM_TIMEOUT == 600
        assert isinstance(constants.DEFAULT_LLM_TIMEOUT, int)
        assert constants.DEFAULT_LLM_TIMEOUT > 0

    def test_default_max_retries(self):
        """Test DEFAULT_MAX_RETRIES constant."""
        assert constants.DEFAULT_MAX_RETRIES == 3
        assert isinstance(constants.DEFAULT_MAX_RETRIES, int)
        assert constants.DEFAULT_MAX_RETRIES > 0

    def test_retry_configuration(self):
        """Test retry configuration constants."""
        assert constants.RETRY_MAX_ATTEMPTS == 5
        assert constants.RETRY_MIN_WAIT == 2
        assert constants.RETRY_MAX_WAIT == 60
        assert constants.RETRY_MULTIPLIER == 1

        assert isinstance(constants.RETRY_MAX_ATTEMPTS, int)
        assert isinstance(constants.RETRY_MIN_WAIT, int)
        assert isinstance(constants.RETRY_MAX_WAIT, int)
        assert isinstance(constants.RETRY_MULTIPLIER, int)

        assert constants.RETRY_MIN_WAIT < constants.RETRY_MAX_WAIT
        assert constants.RETRY_MAX_ATTEMPTS > 0
        assert constants.RETRY_MIN_WAIT > 0
        assert constants.RETRY_MULTIPLIER > 0

    def test_request_id_prefixes(self):
        """Test request ID prefix constants."""
        assert constants.REQUEST_ID_PREFIX_LLM == "llm"
        assert constants.REQUEST_ID_PREFIX_TOOL == "tool"

        assert isinstance(constants.REQUEST_ID_PREFIX_LLM, str)
        assert isinstance(constants.REQUEST_ID_PREFIX_TOOL, str)

        assert len(constants.REQUEST_ID_PREFIX_LLM) > 0
        assert len(constants.REQUEST_ID_PREFIX_TOOL) > 0

    def test_log_content_lengths(self):
        """Test log content length constants."""
        assert constants.LOG_CONTENT_PREVIEW_LENGTH == 200
        assert constants.LOG_RESPONSE_PREVIEW_LENGTH == 500
        assert constants.LOG_ERROR_RESPONSE_LENGTH == 1000

        assert isinstance(constants.LOG_CONTENT_PREVIEW_LENGTH, int)
        assert isinstance(constants.LOG_RESPONSE_PREVIEW_LENGTH, int)
        assert isinstance(constants.LOG_ERROR_RESPONSE_LENGTH, int)

        assert constants.LOG_CONTENT_PREVIEW_LENGTH < constants.LOG_RESPONSE_PREVIEW_LENGTH
        assert constants.LOG_RESPONSE_PREVIEW_LENGTH < constants.LOG_ERROR_RESPONSE_LENGTH
        assert constants.LOG_CONTENT_PREVIEW_LENGTH > 0

    def test_constants_immutability(self):
        """Test that constants maintain their values."""
        original_max_tokens = constants.DEFAULT_MAX_TOKENS
        original_timeout = constants.DEFAULT_LLM_TIMEOUT

        assert constants.DEFAULT_MAX_TOKENS == original_max_tokens
        assert constants.DEFAULT_LLM_TIMEOUT == original_timeout
