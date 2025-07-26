"""Unit tests for app.utils.tool_registry module."""

from unittest.mock import Mock, patch

import pytest

from app.schema.llm.tool import AbstractTool, ToolSchema
from app.utils.tool_registry import ToolRegistry, tool_registry


class MockTool(AbstractTool):
    """Mock tool for testing."""

    tool_schema = Mock(spec=ToolSchema)
    tool_schema.model_dump.return_value = {"name": "MockTool", "description": "Mock tool"}

    @classmethod
    def tool_function(cls):
        """Return mock function."""
        return lambda x: f"MockTool: {x}"


class BrokenTool(AbstractTool):
    """Tool that raises exception during collection."""

    tool_schema = Mock(spec=ToolSchema)

    @classmethod
    def tool_function(cls):
        """Raise exception."""
        raise ValueError("Tool is broken")


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_tool_registry_initialization(self):
        """Test ToolRegistry initializes with empty state."""
        registry = ToolRegistry()

        assert registry._tools is None
        assert registry._initialized is False

    def test_lazy_initialization(self):
        """Test that tools are collected lazily on first access."""
        registry = ToolRegistry()

        assert not registry._initialized

        with patch.object(registry, "_collect_tools") as mock_collect:
            _ = registry.tools
            mock_collect.assert_called_once()
            assert registry._initialized

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_collect_tools_success(self, mock_subclasses):
        """Test successful tool collection."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        registry._tools = {}
        registry._collect_tools()

        assert len(registry._tools) == 1
        assert "MockTool" in registry._tools
        assert registry._tools["MockTool"]["class"] == MockTool
        assert registry._tools["MockTool"]["schema"] == MockTool.tool_schema
        assert callable(registry._tools["MockTool"]["function"])

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    @patch("app.utils.tool_registry.logger")
    def test_collect_tools_with_error(self, mock_logger, mock_subclasses):
        """Test tool collection handles errors gracefully."""
        mock_subclasses.return_value = [MockTool, BrokenTool]

        registry = ToolRegistry()
        registry._tools = {}
        registry._collect_tools()

        assert len(registry._tools) == 1
        assert "MockTool" in registry._tools

        assert "BrokenTool" not in registry._tools

        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args
        assert "Failed to collect tool BrokenTool" in error_call[0][0]

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_get_tool(self, mock_subclasses):
        """Test getting a specific tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        tool = registry.get_tool("MockTool")
        assert tool is not None
        assert tool["class"] == MockTool

        tool = registry.get_tool("NonExistentTool")
        assert tool is None

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_get_tool_schemas(self, mock_subclasses):
        """Test getting tool schemas."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        schemas = registry.get_tool_schemas(["MockTool"])
        assert len(schemas) == 1
        assert schemas[0] == {"name": "MockTool", "description": "Mock tool"}

        MockTool.tool_schema.model_dump.assert_called_with(exclude_none=True)

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_get_tool_schemas_unknown_tool(self, mock_subclasses):
        """Test get_tool_schemas raises error for unknown tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_tool_schemas(["UnknownTool"])

        assert "Unknown tool: UnknownTool" in str(exc_info.value)
        assert "Available tools: ['MockTool']" in str(exc_info.value)

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_get_tool_function(self, mock_subclasses):
        """Test getting tool function."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        func = registry.get_tool_function("MockTool")
        assert callable(func)
        assert func("test") == "MockTool: test"

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_get_tool_function_unknown_tool(self, mock_subclasses):
        """Test get_tool_function raises error for unknown tool."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_tool_function("UnknownTool")

        assert "Tool 'UnknownTool' not found" in str(exc_info.value)
        assert "Available tools: ['MockTool']" in str(exc_info.value)

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    def test_list_tool_names(self, mock_subclasses):
        """Test listing all tool names."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        names = registry.list_tool_names()
        assert names == ["MockTool"]

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    @patch("app.utils.tool_registry.logger")
    def test_reset(self, mock_logger, mock_subclasses):
        """Test resetting the registry."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()

        _ = registry.tools
        assert registry._initialized

        registry.reset()

        assert registry._tools is None
        assert not registry._initialized
        mock_logger.debug.assert_called_with("Tool registry reset")

        _ = registry.tools
        assert registry._initialized

    def test_tools_property_returns_dict(self):
        """Test that tools property returns the internal dict."""
        registry = ToolRegistry()

        with patch.object(registry, "_collect_tools"):
            registry._tools = {"TestTool": {}}
            registry._initialized = True

            tools = registry.tools
            assert tools == {"TestTool": {}}

    @patch("app.utils.tool_registry.AbstractTool.__subclasses__")
    @patch("app.utils.tool_registry.logger")
    def test_logging_during_collection(self, mock_logger, mock_subclasses):
        """Test that tool collection logs appropriately."""
        mock_subclasses.return_value = [MockTool]

        registry = ToolRegistry()
        registry._tools = {}
        registry._collect_tools()

        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Collected tool: MockTool" in call for call in debug_calls)

        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) == 1
        assert "Tool collection completed" in info_calls[0][0][0]
        assert info_calls[0][1]["extra"]["tools_collected"] == 1
        assert info_calls[0][1]["extra"]["tool_names"] == ["MockTool"]


class TestToolRegistrySingleton:
    """Test the tool_registry singleton."""

    def test_tool_registry_singleton_exists(self):
        """Test that tool_registry singleton is available."""
        assert tool_registry is not None
        assert isinstance(tool_registry, ToolRegistry)

    def test_tool_registry_singleton_is_singleton(self):
        """Test that tool_registry is the same instance."""
        from app.utils.tool_registry import tool_registry as registry1
        from app.utils.tool_registry import tool_registry as registry2

        assert registry1 is registry2
