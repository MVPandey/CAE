"""LLM Service for handling OpenAI API interactions."""

import json
import uuid
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion

from ..schema.llm.message import Message, ToolMessage
from ..schema.llm.tool import ToolCall
from ..utils.config import app_settings
from ..utils.constants import (
    DEFAULT_MAX_TOKENS,
    REQUEST_ID_PREFIX_LLM,
)
from ..utils.exceptions import LLMException
from ..utils.json_utils import clean_json_response
from ..utils.logger import logger
from ..utils.tool_registry import tool_registry
from .llm import LLMClient, ToolExecutor


class LLMService:
    """
    Service for interacting with LLM APIs.

    This service provides a high-level interface for:
    - Sending messages to LLM with automatic retry logic
    - Managing tool registration and execution
    - Handling response parsing (text or JSON)
    - Error handling and logging

    Example:
        ```python
        service = LLMService()

        # Simple text query
        response = await service.query_llm(
            Message(role="user", content="Hello, world!")
        )

        # JSON response with tools
        result = await service.query_llm(
            messages=[Message(role="user", content="Get weather")],
            json_response=True,
            tools=["GetWeatherTool"],
            max_tokens=500
        )
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float | None = None,
    ):
        """
        Initialize LLM Service.

        Args:
            base_url: LLM API base URL (defaults to config)
            api_key: LLM API key (defaults to config)
            model_name: Model name to use (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        self.model_name = model_name or app_settings.LLM_MODEL_NAME

        self.client = LLMClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.tool_executor = ToolExecutor()
        self.retry_decorator = LLMClient.get_retry_decorator()

        logger.info(
            "Initialized LLMService",
            extra={
                "model": self.model_name,
                "available_tools": tool_registry.list_tool_names(),
            },
        )

    async def query_llm(
        self,
        messages: Message | list[Message],
        json_response: bool = False,
        tools: str | list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> Message | dict:
        """
        Query the LLM with messages and optional tools.

        Args:
            messages: Single message or list of messages to send to LLM
            json_response: Whether to request JSON formatted response
            tools: Tool name(s) to make available to LLM
            max_tokens: Maximum tokens for the response
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters for the LLM API

        Returns:
            Message response from LLM or parsed JSON if json_response=True

        Raises:
            LLMException: If LLM query fails or response parsing fails
            ValueError: If invalid parameters are provided
        """
        request_id = f"{REQUEST_ID_PREFIX_LLM}_{str(uuid.uuid4())}"

        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {temperature}")

        if top_p is not None and not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")

        logger.info(
            "Starting LLM query",
            extra={
                "request_id": request_id,
                "json_response": json_response,
                "tools_requested": tools,
                "message_count": len(messages) if isinstance(messages, list) else 1,
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

        try:
            normalized_messages = self._normalize_messages(messages)

            prepared_tools = self._prepare_tools(tools) if tools else None

            llm_kwargs = kwargs.copy()
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if top_p is not None:
                llm_kwargs["top_p"] = top_p

            completion = await self._make_llm_request(
                normalized_messages,
                prepared_tools,
                json_response,
                request_id,
                max_tokens=max_tokens,
                **llm_kwargs,
            )

            if completion.choices[0].message.tool_calls:
                completion = await self._handle_tool_workflow(
                    completion,
                    normalized_messages,
                    json_response,
                    request_id,
                    max_tokens=max_tokens,
                    **llm_kwargs,
                )

            return self._process_response(completion, json_response, request_id)

        except LLMException:
            raise
        except Exception as e:
            logger.error(
                "LLM query failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise LLMException(f"Failed to query LLM: {e}") from e

    def _normalize_messages(self, messages: Message | list[Message]) -> list[Message]:
        """
        Convert single message to list format.

        Args:
            messages: Single message or list of messages

        Returns:
            List of messages
        """
        if isinstance(messages, Message):
            return [messages]
        return messages

    def _prepare_tools(self, tools: str | list[str]) -> list[dict]:
        """
        Prepare tool schemas for LLM request.

        Args:
            tools: Tool name(s) to prepare

        Returns:
            List of tool schemas

        Raises:
            ValueError: If unknown tool is requested
        """
        if isinstance(tools, str):
            tools = [tools]

        return tool_registry.get_tool_schemas(tools)

    @property
    def tools(self) -> dict[str, dict[str, Any]]:
        """
        Get available tools from registry.

        Returns:
            Dictionary mapping tool names to their details
        """
        return tool_registry.tools

    async def _make_llm_request(
        self,
        messages: list[Message],
        tools: list[dict] | None,
        json_response: bool,
        request_id: str,
        max_tokens: int,
        **kwargs,
    ) -> ChatCompletion:
        """
        Make the actual LLM API request with retry logic.

        Args:
            messages: List of messages to send
            tools: Tool schemas if any
            json_response: Whether to request JSON response
            request_id: Request ID for logging
            max_tokens: Maximum tokens for response
            **kwargs: Additional LLM parameters

        Returns:
            ChatCompletion from LLM
        """
        message_dicts = [msg.model_dump() for msg in messages]

        request_params = {
            "model": self.model_name,
            "messages": message_dicts,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if tools:
            request_params["tools"] = tools

        if json_response:
            request_params["response_format"] = {"type": "json_object"}

        logger.debug(
            "Making LLM API request",
            extra={
                "request_id": request_id,
                "model": self.model_name,
                "message_count": len(messages),
                "has_tools": bool(tools),
                "json_response": json_response,
                "max_tokens": max_tokens,
            },
        )

        @self.retry_decorator
        async def _make_request():
            client = self.client.get_client()
            return await client.chat.completions.create(**request_params)

        completion = await _make_request()

        logger.info(
            "LLM API request completed",
            extra={
                "request_id": request_id,
                "finish_reason": completion.choices[0].finish_reason,
                "has_tool_calls": bool(completion.choices[0].message.tool_calls),
                "usage": completion.usage.model_dump() if completion.usage else None,
            },
        )

        return completion

    async def _handle_tool_workflow(
        self,
        initial_completion: ChatCompletion,
        messages: list[Message],
        json_response: bool,
        request_id: str,
        max_tokens: int,
        **kwargs,
    ) -> ChatCompletion:
        """
        Handle tool calling workflow.

        Args:
            initial_completion: Initial LLM response with tool calls
            messages: Original messages
            json_response: Whether to request JSON response
            request_id: Request ID for logging
            max_tokens: Maximum tokens for response
            **kwargs: Additional LLM parameters

        Returns:
            Final ChatCompletion after tool execution
        """
        tool_calls = initial_completion.choices[0].message.tool_calls

        logger.info(
            "Processing tool calls",
            extra={
                "request_id": request_id,
                "tool_call_count": len(tool_calls),
            },
        )

        tool_results = await self.tool_executor.execute_tool_calls(tool_calls, execution_id=request_id)

        message_dicts = [msg.model_dump() for msg in messages]
        message_dicts.append(initial_completion.choices[0].message.model_dump())

        for tool_result in tool_results:
            message_dicts.append(tool_result.model_dump())

        request_params = {
            "model": self.model_name,
            "messages": message_dicts,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if json_response:
            request_params["response_format"] = {"type": "json_object"}

        @self.retry_decorator
        async def _make_follow_up():
            client = self.client.get_client()
            return await client.chat.completions.create(**request_params)

        final_completion = await _make_follow_up()

        logger.info(
            "Tool workflow completed",
            extra={
                "request_id": request_id,
                "final_finish_reason": final_completion.choices[0].finish_reason,
            },
        )

        return final_completion

    def _process_response(
        self,
        completion: ChatCompletion,
        json_response: bool,
        request_id: str,
    ) -> Message | dict:
        """
        Process the final LLM response.

        Args:
            completion: ChatCompletion from LLM
            json_response: Whether to parse as JSON
            request_id: Request ID for logging

        Returns:
            Processed response (Message or parsed JSON)

        Raises:
            LLMException: If JSON parsing fails
        """
        response_content = completion.choices[0].message.content

        if json_response:
            if response_content is None:
                raise LLMException("LLM returned None content for JSON response")

            if not response_content.strip():
                raise LLMException("LLM returned empty content for JSON response")

            try:
                parsed_response = json.loads(response_content)
                logger.info(
                    "JSON response parsed successfully",
                    extra={
                        "request_id": request_id,
                        "response_keys": list(parsed_response.keys()) if isinstance(parsed_response, dict) else None,
                    },
                )
                return parsed_response
            except json.JSONDecodeError:
                return clean_json_response(response_content)
        else:
            return completion.choices[0].message

    async def handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        """
        Handle tool calls from the LLM response.

        This method is kept for backward compatibility.
        New code should use tool_executor directly.

        Args:
            tool_calls: List of ToolCall objects from the LLM response

        Returns:
            List of ToolMessage objects with the results
        """
        return await self.tool_executor.execute_tool_calls(tool_calls)

    async def _extract_json_from_response(self, response_text: str) -> dict[str, Any]:
        """
        Extract JSON from LLM response text.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            LLMException: If JSON extraction fails
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return clean_json_response(response_text)

    def _process_tool_calls(self, tool_calls: list[ToolCall], request_id: str = None) -> list[dict[str, Any]]:
        """
        Process tool calls into format suitable for LLM.

        Args:
            tool_calls: List of ToolCall objects

        Returns:
            List of tool call dictionaries
        """
        processed_calls = []
        for tool_call in tool_calls:
            processed_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        return processed_calls
