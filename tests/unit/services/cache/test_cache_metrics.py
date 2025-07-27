"""Tests for cache metrics collection."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.cache.cache_metrics import CacheMetrics, cache_metrics, track_cache_operation


class TestCacheMetrics:
    """Test cache metrics functionality."""

    def test_init(self):
        """Test CacheMetrics initialization."""
        metrics = CacheMetrics()
        assert metrics._operation_timers == {}
        assert metrics._hit_counts == {"hits": 0, "misses": 0}

    @patch("app.services.cache.cache_metrics.cache_operations_total")
    @patch("app.services.cache.cache_metrics.cache_operation_duration_seconds")
    def test_timer_success(self, mock_duration, mock_total):
        """Test timer context manager for successful operations."""
        metrics = CacheMetrics()

        with metrics.timer("get", "redis"):
            pass

        mock_total.labels.assert_called_with(
            operation="get",
            cache_type="redis",
            status="success",
        )
        mock_total.labels.return_value.inc.assert_called_once()
        mock_duration.labels.assert_called_with(
            operation="get",
            cache_type="redis",
        )
        mock_duration.labels.return_value.observe.assert_called_once()

    @patch("app.services.cache.cache_metrics.cache_operations_total")
    @patch("app.services.cache.cache_metrics.cache_operation_duration_seconds")
    def test_timer_error(self, mock_duration, mock_total):
        """Test timer context manager for failed operations."""
        metrics = CacheMetrics()

        with pytest.raises(ValueError):
            with metrics.timer("set", "redis"):
                raise ValueError("Test error")

        mock_total.labels.assert_called_with(
            operation="set",
            cache_type="redis",
            status="error",
        )
        mock_total.labels.return_value.inc.assert_called_once()
        mock_duration.labels.return_value.observe.assert_called_once()

    @patch("app.services.cache.cache_metrics.cache_hit_ratio")
    def test_record_hit(self, mock_ratio):
        """Test recording cache hits."""
        metrics = CacheMetrics()

        metrics.record_hit("semantic")
        assert metrics._hit_counts["hits"] == 1
        assert metrics._hit_counts["misses"] == 0

        mock_ratio.labels.assert_called_with(cache_type="semantic")
        mock_ratio.labels.return_value.set.assert_called_with(1.0)

    @patch("app.services.cache.cache_metrics.cache_hit_ratio")
    def test_record_miss(self, mock_ratio):
        """Test recording cache misses."""
        metrics = CacheMetrics()

        metrics.record_miss("semantic")
        assert metrics._hit_counts["hits"] == 0
        assert metrics._hit_counts["misses"] == 1

        mock_ratio.labels.assert_called_with(cache_type="semantic")
        mock_ratio.labels.return_value.set.assert_called_with(0.0)

    @patch("app.services.cache.cache_metrics.cache_hit_ratio")
    def test_hit_ratio_calculation(self, mock_ratio):
        """Test hit ratio calculation."""
        metrics = CacheMetrics()

        metrics.record_hit("semantic")
        metrics.record_hit("semantic")
        metrics.record_miss("semantic")

        mock_ratio.labels.return_value.set.assert_called_with(2 / 3)

    @patch("app.services.cache.cache_metrics.cache_evictions_total")
    def test_record_eviction(self, mock_evictions):
        """Test recording cache evictions."""
        metrics = CacheMetrics()

        metrics.record_eviction("semantic", "ttl_expired")

        mock_evictions.labels.assert_called_with(
            cache_type="semantic",
            reason="ttl_expired",
        )
        mock_evictions.labels.return_value.inc.assert_called_once()

    @patch("app.services.cache.cache_metrics.cache_size_bytes")
    def test_update_cache_size(self, mock_size):
        """Test updating cache size metric."""
        metrics = CacheMetrics()

        metrics.update_cache_size("semantic", 1024 * 1024)

        mock_size.labels.assert_called_with(cache_type="semantic")
        mock_size.labels.return_value.set.assert_called_with(1024 * 1024)

    @patch("app.services.cache.cache_metrics.cache_entries_total")
    def test_update_entry_count(self, mock_entries):
        """Test updating cache entry count."""
        metrics = CacheMetrics()

        metrics.update_entry_count("semantic", 150)

        mock_entries.labels.assert_called_with(cache_type="semantic")
        mock_entries.labels.return_value.set.assert_called_with(150)

    @patch("app.services.cache.cache_metrics.redis_connections_active")
    def test_update_connection_count(self, mock_connections):
        """Test updating Redis connection count."""
        metrics = CacheMetrics()

        metrics.update_connection_count(5)

        mock_connections.set.assert_called_with(5)

    @patch("app.services.cache.cache_metrics.redis_connection_errors_total")
    def test_record_connection_error(self, mock_errors):
        """Test recording Redis connection errors."""
        metrics = CacheMetrics()

        metrics.record_connection_error("timeout")

        mock_errors.labels.assert_called_with(error_type="timeout")
        mock_errors.labels.return_value.inc.assert_called_once()

    @patch("app.services.cache.cache_metrics.cache_info")
    def test_set_cache_info(self, mock_info):
        """Test setting cache configuration info."""
        metrics = CacheMetrics()

        metrics.set_cache_info(
            ttl_seconds="3600",
            similarity_threshold="0.85",
        )

        mock_info.info.assert_called_with(
            {
                "ttl_seconds": "3600",
                "similarity_threshold": "0.85",
            }
        )


class TestTrackCacheOperationDecorator:
    """Test the track_cache_operation decorator."""

    @pytest.mark.asyncio
    @patch("app.services.cache.cache_metrics.cache_metrics")
    async def test_async_decorator_success(self, mock_metrics):
        """Test decorator with async function that succeeds."""
        mock_timer = MagicMock()
        mock_metrics.timer.return_value = mock_timer
        mock_timer.__enter__ = MagicMock()
        mock_timer.__exit__ = MagicMock(return_value=None)

        @track_cache_operation("get", "redis")
        async def async_func():
            await asyncio.sleep(0.01)
            return "result"

        result = await async_func()

        assert result == "result"
        mock_metrics.timer.assert_called_once_with("get", "redis")
        mock_timer.__enter__.assert_called_once()
        mock_timer.__exit__.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cache.cache_metrics.cache_metrics")
    async def test_async_decorator_error(self, mock_metrics):
        """Test decorator with async function that raises error."""
        mock_timer = MagicMock()
        mock_metrics.timer.return_value = mock_timer
        mock_timer.__enter__ = MagicMock()
        mock_timer.__exit__ = MagicMock(return_value=None)

        @track_cache_operation("set", "redis")
        async def async_func():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await async_func()

        mock_metrics.timer.assert_called_once_with("set", "redis")
        mock_timer.__enter__.assert_called_once()
        mock_timer.__exit__.assert_called_once()

    @patch("app.services.cache.cache_metrics.cache_metrics")
    def test_sync_decorator(self, mock_metrics):
        """Test decorator with sync function."""
        mock_timer = MagicMock()
        mock_metrics.timer.return_value = mock_timer
        mock_timer.__enter__ = MagicMock()
        mock_timer.__exit__ = MagicMock(return_value=None)

        @track_cache_operation("delete", "redis")
        def sync_func():
            return "result"

        result = sync_func()

        assert result == "result"
        mock_metrics.timer.assert_called_once_with("delete", "redis")
        mock_timer.__enter__.assert_called_once()
        mock_timer.__exit__.assert_called_once()


class TestGlobalCacheMetrics:
    """Test the global cache_metrics instance."""

    def test_global_instance_exists(self):
        """Test that global cache_metrics instance exists."""
        assert cache_metrics is not None
        assert isinstance(cache_metrics, CacheMetrics)
