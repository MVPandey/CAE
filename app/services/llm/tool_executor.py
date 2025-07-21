"""Tool execution module for handling LLM tool calls."""

import json
import uuid
from typing import Any

from ...schema.llm.message import ToolMessage
from ...schema.llm.tool import ToolCall
from ...utils.constants import (
    LOG_CONTENT_PREVIEW_LENGTH,
    REQUEST_ID_PREFIX_TOOL,
)
from ...utils.json_utils import safe_json_dumps
from ...utils.logger import logger
from ...utils.tool_registry import tool_registry


class ToolExecutor:
    """Handles execution of LLM tool calls."""

    def __init__(self):
        """Initialize tool executor with tool registry."""
        self.registry = tool_registry

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        execution_id: str | None = None,
    ) -> list[ToolMessage]:
        """
        Execute a list of tool calls from the LLM.

        Args:
            tool_calls: List of ToolCall objects from the LLM response
            execution_id: Optional execution ID for tracking

        Returns:
            List of ToolMessage objects with the results
        """
        if not tool_calls:
            logger.debug("No tool calls to process")
            return []

        execution_id = execution_id or f"{REQUEST_ID_PREFIX_TOOL}_{str(uuid.uuid4())}"

        logger.info(
            "Starting tool execution batch",
            extra={
                "execution_id": execution_id,
                "tool_call_count": len(tool_calls),
                "tool_calls_summary": [
                    {
                        "id": call.id,
                        "function_name": call.function.name,
                        "has_arguments": bool(call.function.arguments),
                    }
                    for call in tool_calls
                ],
            },
        )

        results = []
        successful_calls = 0
        failed_calls = 0

        for call_index, call in enumerate(tool_calls):
            logger.debug(
                "Executing individual tool call",
                extra={
                    "execution_id": execution_id,
                    "call_index": call_index,
                    "tool_call_id": call.id,
                    "function_name": call.function.name,
                    "arguments": call.function.arguments,
                },
            )

            try:
                tool_result = await self._execute_single_tool(call, execution_id, call_index)
                results.append(tool_result)
                successful_calls += 1

                logger.info(
                    "Tool call executed successfully",
                    extra={
                        "execution_id": execution_id,
                        "call_index": call_index,
                        "tool_call_id": call.id,
                        "function_name": call.function.name,
                        "result_preview": self._preview_content(tool_result.content),
                    },
                )

            except Exception as e:
                failed_calls += 1
                error_result = self._create_error_message(call, e, execution_id, call_index)
                results.append(error_result)

        logger.info(
            "Tool execution batch completed",
            extra={
                "execution_id": execution_id,
                "total_calls": len(tool_calls),
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "success_rate": successful_calls / len(tool_calls) if tool_calls else 0,
            },
        )

        return results

    async def _execute_single_tool(self, call: ToolCall, execution_id: str, call_index: int) -> ToolMessage:
        """
        Execute a single tool call.

        Args:
            call: The ToolCall object to execute
            execution_id: Execution batch ID for logging
            call_index: Index of this call in the batch

        Returns:
            ToolMessage with the result

        Raises:
            ValueError: If tool arguments are invalid or tool not found
            Exception: If tool execution fails
        """
        name = call.function.name

        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse tool arguments",
                extra={
                    "execution_id": execution_id,
                    "call_index": call_index,
                    "tool_call_id": call.id,
                    "function_name": name,
                    "raw_arguments": call.function.arguments,
                    "error": str(e),
                },
            )
            raise ValueError(f"Invalid JSON in tool arguments: {e}") from e

        if not isinstance(args, dict):
            raise ValueError(f"Tool arguments must be a dictionary, got {type(args).__name__}")

        try:
            tool_function = self.registry.get_tool_function(name)
        except ValueError:
            logger.error(
                "Tool not found in registry",
                extra={
                    "execution_id": execution_id,
                    "call_index": call_index,
                    "tool_call_id": call.id,
                    "requested_tool": name,
                    "available_tools": self.registry.list_tool_names(),
                },
            )
            raise

        logger.debug(
            "Executing tool function",
            extra={
                "execution_id": execution_id,
                "call_index": call_index,
                "tool_call_id": call.id,
                "function_name": name,
                "parsed_arguments": args,
            },
        )

        try:
            result = await tool_function(**args)

            content = self._serialize_result(result)

            tool_message = ToolMessage(
                role="tool",
                tool_call_id=call.id,
                name=name,
                content=content,
            )

            logger.debug(
                "Tool function executed successfully",
                extra={
                    "execution_id": execution_id,
                    "call_index": call_index,
                    "tool_call_id": call.id,
                    "function_name": name,
                    "result_type": type(result).__name__,
                },
            )

            return tool_message

        except Exception as e:
            logger.error(
                "Tool function execution failed",
                extra={
                    "execution_id": execution_id,
                    "call_index": call_index,
                    "tool_call_id": call.id,
                    "function_name": name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "arguments": args,
                },
                exc_info=True,
            )
            raise

    def _serialize_result(self, result: Any) -> str:
        """
        Serialize tool result to JSON string.

        Args:
            result: Tool execution result

        Returns:
            JSON string representation
        """
        if hasattr(result, "json"):
            return result.json()
        elif hasattr(result, "model_dump"):
            return safe_json_dumps(result.model_dump())
        elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
            return safe_json_dumps(result)
        else:
            return safe_json_dumps(str(result))

    def _create_error_message(
        self, call: ToolCall, error: Exception, execution_id: str, call_index: int
    ) -> ToolMessage:
        """
        Create a ToolMessage for a failed tool call.

        Args:
            call: The failed ToolCall
            error: The exception that occurred
            execution_id: Execution batch ID for logging
            call_index: Index of this call in the batch

        Returns:
            ToolMessage with error information
        """
        logger.error(
            "Creating error tool message",
            extra={
                "execution_id": execution_id,
                "call_index": call_index,
                "tool_call_id": getattr(call, "id", "unknown"),
                "function_name": getattr(call.function, "name", "unknown") if hasattr(call, "function") else "unknown",
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        error_content = {
            "error": str(error),
            "error_type": type(error).__name__,
            "tool_name": getattr(call.function, "name", "unknown") if hasattr(call, "function") else "unknown",
        }

        return ToolMessage(
            role="tool",
            tool_call_id=getattr(call, "id", "unknown"),
            name=getattr(call.function, "name", "unknown") if hasattr(call, "function") else "unknown",
            content=safe_json_dumps(error_content),
        )

    def _preview_content(self, content: Any, max_length: int = LOG_CONTENT_PREVIEW_LENGTH) -> str:
        """
        Create a preview of content for logging.

        Args:
            content: Content to preview
            max_length: Maximum length of preview

        Returns:
            Preview string
        """
        content_str = str(content)
        if len(content_str) > max_length:
            return content_str[:max_length] + "..."
        return content_str
