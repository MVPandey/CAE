"""LLM client management module."""

import openai
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...utils.config import app_settings
from ...utils.constants import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_WAIT,
    RETRY_MIN_WAIT,
    RETRY_MULTIPLIER,
)
from ...utils.logger import logger


class LLMClient:
    """Manages OpenAI client instances with configuration."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize LLM client manager.

        Args:
            base_url: API base URL (defaults to config)
            api_key: API key (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url or app_settings.LLM_API_BASE_URL
        self.api_key = api_key or app_settings.LLM_API_KEY
        self.timeout = timeout or float(app_settings.LLM_TIMEOUT_SECONDS or DEFAULT_LLM_TIMEOUT)
        self.max_retries = max_retries

        logger.debug(
            "Initialized LLM client",
            extra={
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            },
        )

    def get_client(self) -> AsyncOpenAI:
        """
        Get configured AsyncOpenAI client instance.

        Returns:
            Configured AsyncOpenAI client
        """
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @staticmethod
    def _log_retry_attempt(retry_state: RetryCallState) -> None:
        """Log retry attempts for debugging."""
        if retry_state.attempt_number > 1:
            logger.warning(
                "Retrying LLM request",
                extra={
                    "attempt": retry_state.attempt_number,
                    "wait_time": retry_state.next_action.sleep if retry_state.next_action else None,
                    "exception": str(retry_state.outcome.exception()) if retry_state.outcome else None,
                },
            )

    @staticmethod
    def get_retry_decorator():
        """
        Get retry decorator for LLM requests.

        Returns:
            Configured retry decorator
        """
        return retry(
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            retry=(
                retry_if_exception_type(openai.RateLimitError)
                | retry_if_exception_type(openai.APITimeoutError)
                | retry_if_exception_type(openai.APIConnectionError)
            ),
            before_sleep=LLMClient._log_retry_attempt,
        )
