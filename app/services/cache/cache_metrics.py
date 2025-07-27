"""Cache-specific metrics using the centralized metrics system."""

import asyncio
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, Info

from ...utils.logger import logger
from ...utils.metrics import REGISTRY

cache_operations_total = Counter(
    "cache_operations_total",
    "Total number of cache operations",
    ["operation", "cache_type", "status"],
    registry=REGISTRY,
)

cache_operation_duration_seconds = Histogram(
    "cache_operation_duration_seconds",
    "Duration of cache operations in seconds",
    ["operation", "cache_type"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=REGISTRY,
)

cache_hit_ratio = Gauge(
    "cache_hit_ratio",
    "Cache hit ratio (0-1)",
    ["cache_type"],
    registry=REGISTRY,
)

cache_size_bytes = Gauge(
    "cache_size_bytes",
    "Estimated cache size in bytes",
    ["cache_type"],
    registry=REGISTRY,
)

cache_entries_total = Gauge(
    "cache_entries_total",
    "Total number of cache entries",
    ["cache_type"],
    registry=REGISTRY,
)

redis_connections_active = Gauge(
    "redis_connections_active",
    "Number of active Redis connections",
    registry=REGISTRY,
)

redis_connection_errors_total = Counter(
    "redis_connection_errors_total",
    "Total number of Redis connection errors",
    ["error_type"],
    registry=REGISTRY,
)

cache_evictions_total = Counter(
    "cache_evictions_total",
    "Total number of cache evictions",
    ["cache_type", "reason"],
    registry=REGISTRY,
)

cache_info = Info(
    "cache_config",
    "Cache configuration information",
    registry=REGISTRY,
)


class CacheMetrics:
    """Centralized cache metrics collection."""

    def __init__(self):
        self._operation_timers = {}
        self._hit_counts = {"hits": 0, "misses": 0}

    @contextmanager
    def timer(self, operation: str, cache_type: str = "redis"):
        """Context manager for timing cache operations."""
        start_time = time.time()
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            logger.error(f"Cache operation {operation} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            cache_operations_total.labels(
                operation=operation,
                cache_type=cache_type,
                status=status,
            ).inc()
            cache_operation_duration_seconds.labels(
                operation=operation,
                cache_type=cache_type,
            ).observe(duration)

    def record_hit(self, cache_type: str = "semantic"):
        """Record a cache hit."""
        self._hit_counts["hits"] += 1
        self._update_hit_ratio(cache_type)

    def record_miss(self, cache_type: str = "semantic"):
        """Record a cache miss."""
        self._hit_counts["misses"] += 1
        self._update_hit_ratio(cache_type)

    def _update_hit_ratio(self, cache_type: str):
        """Update the hit ratio metric."""
        total = self._hit_counts["hits"] + self._hit_counts["misses"]
        if total > 0:
            ratio = self._hit_counts["hits"] / total
            cache_hit_ratio.labels(cache_type=cache_type).set(ratio)

    def record_eviction(self, cache_type: str, reason: str):
        """Record a cache eviction."""
        cache_evictions_total.labels(
            cache_type=cache_type,
            reason=reason,
        ).inc()

    def update_cache_size(self, cache_type: str, size_bytes: int):
        """Update cache size metric."""
        cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)

    def update_entry_count(self, cache_type: str, count: int):
        """Update cache entry count."""
        cache_entries_total.labels(cache_type=cache_type).set(count)

    def update_connection_count(self, count: int):
        """Update active Redis connection count."""
        redis_connections_active.set(count)

    def record_connection_error(self, error_type: str):
        """Record a Redis connection error."""
        redis_connection_errors_total.labels(error_type=error_type).inc()

    def set_cache_info(self, **kwargs):
        """Set cache configuration info."""
        cache_info.info(kwargs)


cache_metrics = CacheMetrics()


def track_cache_operation(operation: str, cache_type: str = "redis"):
    """Decorator to track cache operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with cache_metrics.timer(operation, cache_type):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with cache_metrics.timer(operation, cache_type):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
