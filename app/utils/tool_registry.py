"""Tool registry for collecting and managing LLM tools."""

from typing import Any, Callable

from ..schema.llm.tool import AbstractTool
from ..utils.logger import logger


class ToolRegistry:
    """Registry for managing LLM tools."""

    def __init__(self):
        """Initialize empty registry. Tools will be collected on first access."""
        self._tools: dict[str, dict[str, Any]] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure tools are collected. This is done lazily to avoid circular imports."""
        if not self._initialized:
            self._tools = {}
            self._collect_tools()
            self._initialized = True

    def _collect_tools(self) -> None:
        """
        Collects all AbstractTool subclasses and their schemas/functions.
        """
        for tool_class in AbstractTool.__subclasses__():
            try:
                tool_name = tool_class.__name__

                self._tools[tool_name] = {
                    "class": tool_class,
                    "schema": tool_class.tool_schema,
                    "function": tool_class.tool_function(),
                }

                logger.debug(f"Collected tool: {tool_name}")
            except Exception as e:
                logger.error(
                    f"Failed to collect tool {tool_class.__name__}",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                continue

        logger.info(
            "Tool collection completed",
            extra={
                "tools_collected": len(self._tools),
                "tool_names": list(self._tools.keys()),
            },
        )

    @property
    def tools(self) -> dict[str, dict[str, Any]]:
        """Get all registered tools."""
        self._ensure_initialized()
        return self._tools

    def get_tool(self, name: str) -> dict[str, Any] | None:
        """
        Get a specific tool by name.

        Args:
            name: Tool name

        Returns:
            Tool details or None if not found
        """
        self._ensure_initialized()
        return self._tools.get(name)

    def get_tool_schemas(self, tool_names: list[str]) -> list[dict]:
        """
        Get schemas for specified tools.

        Args:
            tool_names: List of tool names

        Returns:
            List of tool schemas

        Raises:
            ValueError: If any tool is not found
        """
        self._ensure_initialized()
        schemas = []

        for name in tool_names:
            if name not in self._tools:
                available = list(self._tools.keys())
                raise ValueError(f"Unknown tool: {name}. Available tools: {available}")

            tool_schema = self._tools[name]["schema"].model_dump(exclude_none=True)
            schemas.append(tool_schema)

        return schemas

    def get_tool_function(self, name: str) -> Callable:
        """
        Get the function for a specific tool.

        Args:
            name: Tool name

        Returns:
            Tool function

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            available = list(self.list_tool_names())
            raise ValueError(f"Tool '{name}' not found. Available tools: {available}")
        return tool["function"]

    def list_tool_names(self) -> list[str]:
        """
        Get list of all available tool names.

        Returns:
            List of tool names
        """
        self._ensure_initialized()
        return list(self._tools.keys())

    def reset(self) -> None:
        """Reset the registry, forcing re-collection on next access."""
        self._tools = None
        self._initialized = False
        logger.debug("Tool registry reset")


tool_registry = ToolRegistry()
