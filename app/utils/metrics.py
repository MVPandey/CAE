"""Centralized metrics system for the entire application using Prometheus."""

import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
from prometheus_client.core import CollectorRegistry

from .config import app_settings
from .logger import logger

REGISTRY = CollectorRegistry()

app_info = Info(
    "app_info",
    "Application information",
    registry=REGISTRY,
)

request_total = Counter(
    "app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

request_duration_seconds = Histogram(
    "app_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

active_requests = Gauge(
    "app_active_requests",
    "Number of active requests",
    ["method", "endpoint"],
    registry=REGISTRY,
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM API requests",
    ["model", "operation", "status"],
    registry=REGISTRY,
)

llm_tokens_used = Counter(
    "llm_tokens_used_total",
    "Total number of tokens used",
    ["model", "operation", "token_type"],
    registry=REGISTRY,
)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    ["model", "operation"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

llm_cost_dollars = Counter(
    "llm_cost_dollars_total",
    "Total LLM API cost in dollars",
    ["model", "operation"],
    registry=REGISTRY,
)

mcts_runs_total = Counter(
    "mcts_runs_total",
    "Total number of MCTS runs",
    ["status"],
    registry=REGISTRY,
)

mcts_nodes_explored = Histogram(
    "mcts_nodes_explored",
    "Number of nodes explored per MCTS run",
    buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
    registry=REGISTRY,
)

mcts_tree_depth = Histogram(
    "mcts_tree_depth",
    "Maximum tree depth reached",
    buckets=(1, 2, 3, 5, 10, 15, 20, 30, 50),
    registry=REGISTRY,
)

mcts_run_duration = Histogram(
    "mcts_run_duration_seconds",
    "MCTS run duration",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    ["query_type", "table"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=REGISTRY,
)

db_connections_active = Gauge(
    "db_connections_active",
    "Number of active database connections",
    registry=REGISTRY,
)

mcp_tool_calls_total = Counter(
    "mcp_tool_calls_total",
    "Total number of MCP tool calls",
    ["tool_name", "status"],
    registry=REGISTRY,
)

mcp_tool_duration = Histogram(
    "mcp_tool_duration_seconds",
    "MCP tool execution duration",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

mcp_active_sessions = Gauge(
    "mcp_active_sessions",
    "Number of active MCP sessions",
    registry=REGISTRY,
)


class MetricsCollector:
    """Central metrics collector for the application."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.registry = REGISTRY
        self._initialized = False

    def initialize(self):
        """Initialize application metrics."""
        if self._initialized:
            return

        app_info.info(
            {
                "version": "0.0.1",
                "environment": app_settings.LOG_LEVEL,
                "redis_host": app_settings.REDIS_HOST,
                "db_host": app_settings.DB_HOST,
            }
        )
        self._initialized = True
        logger.info("Metrics collector initialized")

    @contextmanager
    def timer(self, metric: Histogram, **labels):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            metric.labels(**labels).observe(duration)

    def track_request(self, method: str, endpoint: str):
        """Decorator to track HTTP requests."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                active_requests.labels(method=method, endpoint=endpoint).inc()
                start_time = time.time()
                status = "success"
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    request_total.labels(method=method, endpoint=endpoint, status=status).inc()
                    request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
                    active_requests.labels(method=method, endpoint=endpoint).dec()

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                active_requests.labels(method=method, endpoint=endpoint).inc()
                start_time = time.time()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    request_total.labels(method=method, endpoint=endpoint, status=status).inc()
                    request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
                    active_requests.labels(method=method, endpoint=endpoint).dec()

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def track_llm_request(
        self,
        model: str,
        operation: str,
        tokens_used: Optional[dict[str, int]] = None,
        cost: Optional[float] = None,
        duration: Optional[float] = None,
        status: str = "success",
    ):
        """Track LLM API request metrics."""
        llm_requests_total.labels(model=model, operation=operation, status=status).inc()

        if tokens_used:
            for token_type, count in tokens_used.items():
                llm_tokens_used.labels(model=model, operation=operation, token_type=token_type).add(count)

        if cost is not None:
            llm_cost_dollars.labels(model=model, operation=operation).add(cost)

        if duration is not None:
            llm_request_duration.labels(model=model, operation=operation).observe(duration)

    def track_mcts_run(
        self,
        nodes_explored: int,
        tree_depth: int,
        duration: float,
        status: str = "success",
    ):
        """Track MCTS run metrics."""
        mcts_runs_total.labels(status=status).inc()
        mcts_nodes_explored.observe(nodes_explored)
        mcts_tree_depth.observe(tree_depth)
        mcts_run_duration.observe(duration)

    def track_mcp_tool_call(self, tool_name: str):
        """Decorator to track MCP tool calls."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    logger.error(f"MCP tool {tool_name} failed: {e}")
                    raise
                finally:
                    duration = time.time() - start_time
                    mcp_tool_calls_total.labels(tool_name=tool_name, status=status).inc()
                    mcp_tool_duration.labels(tool_name=tool_name).observe(duration)

            return async_wrapper

        return decorator

    def update_mcp_sessions(self, count: int):
        """Update active MCP sessions count."""
        mcp_active_sessions.set(count)

    def track_db_query(self, query_type: str, table: str):
        """Decorator to track database queries."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.timer(db_query_duration, query_type=query_type, table=table):
                    return await func(*args, **kwargs)

            return async_wrapper

        return decorator

    def update_db_connections(self, count: int):
        """Update active database connections."""
        db_connections_active.set(count)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as a dictionary for monitoring endpoints."""
        metrics = {}

        for collector in self.registry.collect():
            for metric in collector.collect():
                metric_type = metric.type

                if metric_type in ["counter", "gauge"]:
                    for sample in metric.samples:
                        key = f"{sample.name}"
                        if sample.labels:
                            label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
                            key = f"{key}{{{label_str}}}"
                        metrics[key] = sample.value
                elif metric_type == "histogram":
                    for sample in metric.samples:
                        if sample.name.endswith("_count") or sample.name.endswith("_sum"):
                            key = sample.name
                            if sample.labels:
                                label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
                                key = f"{key}{{{label_str}}}"
                            metrics[key] = sample.value
                elif metric_type == "summary":
                    for sample in metric.samples:
                        key = sample.name
                        if sample.labels:
                            label_str = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
                            key = f"{key}{{{label_str}}}"
                        metrics[key] = sample.value

        return metrics


metrics_collector = MetricsCollector()


def track_request(method: str, endpoint: str):
    """Decorator to track HTTP requests."""
    return metrics_collector.track_request(method, endpoint)


def track_mcp_tool(tool_name: str):
    """Decorator to track MCP tool calls."""
    return metrics_collector.track_mcp_tool_call(tool_name)


def track_db_query(query_type: str, table: str):
    """Decorator to track database queries."""
    return metrics_collector.track_db_query(query_type, table)
