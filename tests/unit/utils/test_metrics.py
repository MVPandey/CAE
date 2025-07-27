"""Tests for the centralized metrics system."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.utils.metrics import (
    MetricsCollector,
    metrics_collector,
    track_db_query,
    track_mcp_tool,
    track_request,
)


class TestMetricsCollector:
    """Test the MetricsCollector class."""

    def test_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector.registry is not None
        assert collector._initialized is False

    @patch("app.utils.metrics.app_info")
    def test_initialize(self, mock_info):
        """Test initializing metrics."""
        collector = MetricsCollector()

        collector.initialize()

        assert collector._initialized is True
        mock_info.info.assert_called_once()

        collector.initialize()
        assert mock_info.info.call_count == 1

    def test_timer(self):
        """Test timer context manager."""
        collector = MetricsCollector()
        mock_histogram = MagicMock()

        with collector.timer(mock_histogram, operation="test", cache_type="redis"):
            time.sleep(0.01)

        mock_histogram.labels.assert_called_with(operation="test", cache_type="redis")
        mock_histogram.labels.return_value.observe.assert_called_once()

        observed_duration = mock_histogram.labels.return_value.observe.call_args[0][0]
        assert observed_duration > 0.01

    @pytest.mark.asyncio
    @patch("app.utils.metrics.request_total")
    @patch("app.utils.metrics.request_duration_seconds")
    @patch("app.utils.metrics.active_requests")
    async def test_track_request_async_success(self, mock_active, mock_duration, mock_total):
        """Test tracking async request that succeeds."""
        collector = MetricsCollector()

        @collector.track_request("GET", "/test")
        async def test_endpoint():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_endpoint()

        assert result == "result"

        mock_active.labels.assert_called_with(method="GET", endpoint="/test")
        assert mock_active.labels.return_value.inc.call_count == 1
        assert mock_active.labels.return_value.dec.call_count == 1

        mock_total.labels.assert_called_with(method="GET", endpoint="/test", status="success")
        mock_total.labels.return_value.inc.assert_called_once()

        mock_duration.labels.assert_called_with(method="GET", endpoint="/test")
        mock_duration.labels.return_value.observe.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.utils.metrics.request_total")
    @patch("app.utils.metrics.request_duration_seconds")
    @patch("app.utils.metrics.active_requests")
    async def test_track_request_async_error(self, mock_active, mock_duration, mock_total):
        """Test tracking async request that fails."""
        collector = MetricsCollector()

        @collector.track_request("POST", "/test")
        async def test_endpoint():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await test_endpoint()

        mock_total.labels.assert_called_with(method="POST", endpoint="/test", status="error")
        mock_total.labels.return_value.inc.assert_called_once()

        assert mock_active.labels.return_value.dec.call_count == 1

    @patch("app.utils.metrics.request_total")
    @patch("app.utils.metrics.request_duration_seconds")
    @patch("app.utils.metrics.active_requests")
    def test_track_request_sync(self, mock_active, mock_duration, mock_total):
        """Test tracking sync request."""
        collector = MetricsCollector()

        @collector.track_request("GET", "/sync")
        def test_endpoint():
            return "sync result"

        result = test_endpoint()

        assert result == "sync result"

        mock_total.labels.assert_called_with(method="GET", endpoint="/sync", status="success")
        mock_total.labels.return_value.inc.assert_called_once()

    @patch("app.utils.metrics.llm_requests_total")
    @patch("app.utils.metrics.llm_tokens_used")
    @patch("app.utils.metrics.llm_cost_dollars")
    @patch("app.utils.metrics.llm_request_duration")
    def test_track_llm_request(self, mock_duration, mock_cost, mock_tokens, mock_total):
        """Test tracking LLM request metrics."""
        collector = MetricsCollector()

        collector.track_llm_request(
            model="gpt-4",
            operation="completion",
            tokens_used={"prompt": 100, "completion": 50},
            cost=0.0075,
            duration=2.5,
            status="success",
        )

        mock_total.labels.assert_called_with(model="gpt-4", operation="completion", status="success")
        mock_total.labels.return_value.inc.assert_called_once()

        assert mock_tokens.labels.call_count == 2
        mock_tokens.labels.assert_any_call(model="gpt-4", operation="completion", token_type="prompt")
        mock_tokens.labels.assert_any_call(model="gpt-4", operation="completion", token_type="completion")

        mock_cost.labels.assert_called_with(model="gpt-4", operation="completion")
        mock_cost.labels.return_value.add.assert_called_with(0.0075)

        mock_duration.labels.assert_called_with(model="gpt-4", operation="completion")
        mock_duration.labels.return_value.observe.assert_called_with(2.5)

    @patch("app.utils.metrics.mcts_runs_total")
    @patch("app.utils.metrics.mcts_nodes_explored")
    @patch("app.utils.metrics.mcts_tree_depth")
    @patch("app.utils.metrics.mcts_run_duration")
    def test_track_mcts_run(self, mock_duration, mock_depth, mock_nodes, mock_total):
        """Test tracking MCTS run metrics."""
        collector = MetricsCollector()

        collector.track_mcts_run(
            nodes_explored=150,
            tree_depth=8,
            duration=15.5,
            status="success",
        )

        mock_total.labels.assert_called_with(status="success")
        mock_total.labels.return_value.inc.assert_called_once()

        mock_nodes.observe.assert_called_with(150)
        mock_depth.observe.assert_called_with(8)
        mock_duration.observe.assert_called_with(15.5)

    @pytest.mark.asyncio
    @patch("app.utils.metrics.mcp_tool_calls_total")
    @patch("app.utils.metrics.mcp_tool_duration")
    async def test_track_mcp_tool_call_success(self, mock_duration, mock_total):
        """Test tracking MCP tool call that succeeds."""
        collector = MetricsCollector()

        @collector.track_mcp_tool_call("analyze")
        async def analyze_tool():
            await asyncio.sleep(0.01)
            return {"result": "success"}

        result = await analyze_tool()

        assert result == {"result": "success"}

        mock_total.labels.assert_called_with(tool_name="analyze", status="success")
        mock_total.labels.return_value.inc.assert_called_once()

        mock_duration.labels.assert_called_with(tool_name="analyze")
        mock_duration.labels.return_value.observe.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.utils.metrics.mcp_tool_calls_total")
    @patch("app.utils.metrics.mcp_tool_duration")
    async def test_track_mcp_tool_call_error(self, mock_duration, mock_total):
        """Test tracking MCP tool call that fails."""
        collector = MetricsCollector()

        @collector.track_mcp_tool_call("failing_tool")
        async def failing_tool():
            await asyncio.sleep(0.01)
            raise RuntimeError("Tool failed")

        with pytest.raises(RuntimeError, match="Tool failed"):
            await failing_tool()

        mock_total.labels.assert_called_with(tool_name="failing_tool", status="error")
        mock_total.labels.return_value.inc.assert_called_once()

    @patch("app.utils.metrics.mcp_active_sessions")
    def test_update_mcp_sessions(self, mock_sessions):
        """Test updating MCP session count."""
        collector = MetricsCollector()

        collector.update_mcp_sessions(3)

        mock_sessions.set.assert_called_with(3)

    @pytest.mark.asyncio
    @patch("app.utils.metrics.db_query_duration")
    async def test_track_db_query(self, mock_duration):
        """Test tracking database query."""
        collector = MetricsCollector()

        mock_histogram = MagicMock()
        mock_duration.labels.return_value = mock_histogram

        @collector.track_db_query("select", "users")
        async def query_users():
            await asyncio.sleep(0.01)
            return ["user1", "user2"]

        result = await query_users()

        assert result == ["user1", "user2"]
        mock_duration.labels.assert_called_with(query_type="select", table="users")

    @patch("app.utils.metrics.db_connections_active")
    def test_update_db_connections(self, mock_connections):
        """Test updating database connection count."""
        collector = MetricsCollector()

        collector.update_db_connections(10)

        mock_connections.set.assert_called_with(10)

    def test_get_metrics(self):
        """Test getting Prometheus metrics."""
        collector = MetricsCollector()

        metrics_text = collector.get_metrics()

        assert isinstance(metrics_text, bytes)
        assert len(metrics_text) > 0

    @patch("app.utils.metrics.REGISTRY")
    def test_get_metrics_dict(self, mock_registry):
        """Test getting metrics as dictionary."""
        mock_metric = MagicMock()
        mock_sample = MagicMock()
        mock_sample.name = "test_metric"
        mock_sample.labels = {"label1": "value1"}
        mock_sample.value = 42.0

        mock_metric.name = "test_metric"
        mock_metric.type = "counter"
        mock_metric.samples = [mock_sample]

        mock_collector = MagicMock()
        mock_collector.collect.return_value = [mock_metric]

        mock_registry.collect.return_value = [mock_collector]

        collector = MetricsCollector()
        collector.registry = mock_registry

        metrics_dict = collector.get_metrics_dict()

        assert isinstance(metrics_dict, dict)
        assert 'test_metric{label1="value1"}' in metrics_dict
        assert metrics_dict['test_metric{label1="value1"}'] == 42.0


class TestGlobalMetricsCollector:
    """Test the global metrics_collector instance."""

    def test_global_instance_exists(self):
        """Test that global metrics_collector instance exists."""
        assert metrics_collector is not None
        assert isinstance(metrics_collector, MetricsCollector)


class TestConvenienceDecorators:
    """Test the convenience decorator functions."""

    @pytest.mark.asyncio
    @patch("app.utils.metrics.metrics_collector")
    async def test_track_request_decorator(self, mock_collector):
        """Test track_request decorator function."""
        mock_decorator = MagicMock()
        mock_collector.track_request.return_value = mock_decorator

        @track_request("GET", "/api/test")
        async def test_endpoint():
            return "result"

        mock_collector.track_request.assert_called_with("GET", "/api/test")

    @pytest.mark.asyncio
    @patch("app.utils.metrics.metrics_collector")
    async def test_track_mcp_tool_decorator(self, mock_collector):
        """Test track_mcp_tool decorator function."""
        mock_decorator = MagicMock()
        mock_collector.track_mcp_tool_call.return_value = mock_decorator

        @track_mcp_tool("test_tool")
        async def test_tool():
            return "result"

        mock_collector.track_mcp_tool_call.assert_called_with("test_tool")

    @pytest.mark.asyncio
    @patch("app.utils.metrics.metrics_collector")
    async def test_track_db_query_decorator(self, mock_collector):
        """Test track_db_query decorator function."""
        mock_decorator = MagicMock()
        mock_collector.track_db_query.return_value = mock_decorator

        @track_db_query("insert", "logs")
        async def insert_log():
            return True

        mock_collector.track_db_query.assert_called_with("insert", "logs")
