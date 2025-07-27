"""Tests for monitoring API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMetricsEndpoints:
    """Test metrics endpoints."""

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_get_metrics_prometheus(self, mock_collector, async_client):
        """Test getting metrics in Prometheus format."""
        mock_collector.get_metrics.return_value = (
            b"# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 42.0\n"
        )

        response = await async_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert b"test_metric" in response.content
        mock_collector.get_metrics.assert_called_once()

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_get_metrics_json(self, mock_collector, async_client):
        """Test getting metrics in JSON format."""
        mock_metrics = {
            "test_counter": 42.0,
            'test_gauge{label="value"}': 3.14,
            "test_histogram_count": 100,
            "test_histogram_sum": 250.5,
        }
        mock_collector.get_metrics_dict.return_value = mock_metrics

        response = await async_client.get("/metrics/json")

        assert response.status_code == 200
        assert response.json() == mock_metrics
        mock_collector.get_metrics_dict.assert_called_once()

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_get_metrics_empty(self, mock_collector, async_client):
        """Test getting metrics when no metrics exist."""
        mock_collector.get_metrics.return_value = b""
        mock_collector.get_metrics_dict.return_value = {}

        response = await async_client.get("/metrics")
        assert response.status_code == 200
        assert response.content == b""

        response = await async_client.get("/metrics/json")
        assert response.status_code == 200
        assert response.json() == {}

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_get_metrics_exception_handling(self, mock_collector, async_client):
        """Test error handling in metrics endpoints."""
        mock_collector.get_metrics.side_effect = Exception("Metrics error")
        mock_collector.get_metrics_dict.side_effect = Exception("Metrics error")

        response = await async_client.get("/metrics")
        assert response.status_code == 500

        response = await async_client.get("/metrics/json")
        assert response.status_code == 500


class TestHealthEndpoints:
    """Test health check endpoints."""

    @patch("app.services.cache.redis_manager.redis_manager")
    @patch("app.services.cache.semantic_cache.semantic_cache")
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_cache, mock_redis, async_client):
        """Test health check when all services are healthy."""
        mock_redis.is_healthy = True

        async def mock_health_check():
            return True

        mock_cache.health_check = mock_health_check

        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["redis"]["status"] == "healthy"
        assert data["services"]["cache"]["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    @patch("app.services.cache.redis_manager.redis_manager")
    @patch("app.services.cache.semantic_cache.semantic_cache")
    @pytest.mark.asyncio
    async def test_health_check_redis_unhealthy(self, mock_cache, mock_redis, async_client):
        """Test health check when Redis is unhealthy."""
        mock_redis.is_healthy = False

        async def mock_health_check():
            return True

        mock_cache.health_check = mock_health_check

        response = await async_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["redis"]["status"] == "unhealthy"

    @patch("app.services.cache.redis_manager.redis_manager")
    @patch("app.services.cache.semantic_cache.semantic_cache")
    @pytest.mark.asyncio
    async def test_health_check_cache_unhealthy(self, mock_cache, mock_redis, async_client):
        """Test health check when cache is unhealthy."""
        mock_redis.is_healthy = True

        async def mock_health_check():
            return False

        mock_cache.health_check = mock_health_check

        response = await async_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["cache"]["status"] == "unhealthy"
        assert "error" in data["services"]["cache"]

    @patch("app.services.cache.redis_manager.redis_manager")
    @patch("app.services.cache.semantic_cache.semantic_cache")
    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_cache, mock_redis, async_client):
        """Test health check when an exception occurs."""
        from unittest.mock import PropertyMock

        type(mock_redis).is_healthy = PropertyMock(side_effect=Exception("Redis connection error"))

        response = await async_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data["services"]["redis"]
        assert "Redis connection error" in data["services"]["redis"]["error"]

    @patch("app.services.cache.redis_manager.redis_manager")
    @patch("app.services.cache.semantic_cache.semantic_cache")
    @pytest.mark.asyncio
    async def test_health_check_detailed(self, mock_cache, mock_redis, async_client):
        """Test detailed health check endpoint."""
        mock_redis.is_healthy = True
        mock_redis.get_connection_info = MagicMock(
            return_value={
                "connected": True,
                "pool_size": 20,
                "active_connections": 5,
            }
        )

        async def mock_health_check():
            return True

        mock_cache.health_check = mock_health_check

        async def mock_get_stats():
            return {
                "total_entries": 100,
                "memory_usage": 1024 * 1024,
                "hit_rate": 0.85,
            }

        mock_cache.get_stats = mock_get_stats

        response = await async_client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "connection_info" in data["services"]["redis"]
        assert "stats" in data["services"]["cache"]


class TestCacheMonitoringEndpoints:
    """Test cache-specific monitoring endpoints."""

    @patch("app.api.monitoring.redis_manager")
    @patch("app.api.monitoring.semantic_cache")
    @patch("app.api.monitoring.embedding_service")
    @pytest.mark.asyncio
    async def test_get_cache_statistics_success(self, mock_embedding, mock_semantic, mock_redis, async_client):
        """Test successful cache statistics retrieval."""
        mock_redis.get_info = AsyncMock(return_value={"status": "healthy", "version": "7.0"})
        mock_semantic.get_stats = MagicMock(
            return_value={
                "hit_rate": 0.85,
                "total_requests": 100,
                "exact_hits": 50,
                "similarity_hits": 35,
                "misses": 15,
            }
        )
        mock_embedding.get_stats = MagicMock(
            return_value={
                "embeddings_cached": 50,
                "cache_hits": 40,
                "cache_misses": 10,
            }
        )

        response = await async_client.get("/monitoring/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "redis" in data
        assert "semantic_cache" in data
        assert "embeddings" in data
        assert "recommendations" in data

    @patch("app.api.monitoring.redis_manager")
    @patch("app.api.monitoring.semantic_cache")
    @patch("app.api.monitoring.embedding_service")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_get_cache_statistics_error(
        self, mock_logger, mock_embedding, mock_semantic, mock_redis, async_client
    ):
        """Test cache statistics error handling."""
        mock_redis.get_info.side_effect = Exception("Database error")

        response = await async_client.get("/monitoring/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data
        mock_logger.error.assert_called_once()

    @patch("app.api.monitoring.redis_manager")
    @pytest.mark.asyncio
    async def test_check_cache_health_healthy(self, mock_redis, async_client):
        """Test cache health check when healthy."""
        mock_redis.exists = AsyncMock(return_value=True)
        mock_redis._is_healthy = True

        response = await async_client.get("/monitoring/cache/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["redis"] == "healthy"

    @patch("app.api.monitoring.redis_manager")
    @pytest.mark.asyncio
    async def test_check_cache_health_unhealthy(self, mock_redis, async_client):
        """Test cache health check when unhealthy."""
        mock_redis.exists = AsyncMock(return_value=False)
        mock_redis._is_healthy = False

        response = await async_client.get("/monitoring/cache/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["redis"] == "unhealthy"

    @patch("app.api.monitoring.redis_manager")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_check_cache_health_error(self, mock_logger, mock_redis, async_client):
        """Test cache health check error handling."""
        mock_redis.exists.side_effect = Exception("Connection error")

        response = await async_client.get("/monitoring/cache/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data
        mock_logger.error.assert_called_once()

    @patch("app.api.monitoring.semantic_cache")
    @patch("app.api.monitoring.embedding_service")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_clear_cache_success(self, mock_logger, mock_embedding, mock_semantic, async_client):
        """Test successful cache clearing."""
        mock_semantic.clear_all = AsyncMock(return_value=25)
        mock_embedding.clear_cache = AsyncMock(return_value=15)

        response = await async_client.delete("/monitoring/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["semantic_entries_cleared"] == 25
        assert data["embedding_entries_cleared"] == 15
        assert data["total_cleared"] == 40
        mock_logger.info.assert_called_once()

    @patch("app.api.monitoring.semantic_cache")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_clear_cache_error(self, mock_logger, mock_semantic, async_client):
        """Test cache clearing error handling."""
        mock_semantic.clear_all.side_effect = Exception("Clear failed")

        response = await async_client.delete("/monitoring/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data
        mock_logger.error.assert_called_once()

    @patch("app.api.monitoring.semantic_cache")
    @pytest.mark.asyncio
    async def test_cache_recommendations_low_hit_rate(self, mock_semantic, async_client):
        """Test cache recommendations for low hit rate."""
        mock_semantic.get_stats = MagicMock(
            return_value={
                "hit_rate": 0.15,
                "total_requests": 100,
                "exact_hits": 10,
                "similarity_hits": 5,
                "misses": 85,
            }
        )

        with patch("app.api.monitoring.redis_manager") as mock_redis:
            mock_redis.get_info = AsyncMock(return_value={"status": "healthy"})
            with patch("app.api.monitoring.embedding_service") as mock_embedding:
                mock_embedding.get_stats = MagicMock(return_value={"cache_hits": 10})

                response = await async_client.get("/monitoring/cache/stats")
                data = response.json()
                assert any("Low cache hit rate" in rec for rec in data["recommendations"])

    @patch("app.api.monitoring.semantic_cache")
    @pytest.mark.asyncio
    async def test_cache_recommendations_high_hit_rate(self, mock_semantic, async_client):
        """Test cache recommendations for high hit rate."""
        mock_semantic.get_stats = MagicMock(
            return_value={
                "hit_rate": 0.95,
                "total_requests": 100,
                "exact_hits": 90,
                "similarity_hits": 5,
                "misses": 5,
            }
        )

        with patch("app.api.monitoring.redis_manager") as mock_redis:
            mock_redis.get_info = AsyncMock(return_value={"status": "healthy"})
            with patch("app.api.monitoring.embedding_service") as mock_embedding:
                mock_embedding.get_stats = MagicMock(return_value={"cache_hits": 90})

                response = await async_client.get("/monitoring/cache/stats")
                data = response.json()
                assert any("Very high cache hit rate" in rec for rec in data["recommendations"])

    @patch("app.api.monitoring.semantic_cache")
    @pytest.mark.asyncio
    async def test_cache_recommendations_high_usage(self, mock_semantic, async_client):
        """Test cache recommendations for high usage."""
        mock_semantic.get_stats = MagicMock(
            return_value={
                "hit_rate": 0.5,
                "total_requests": 15000,
                "exact_hits": 7500,
                "similarity_hits": 0,
                "misses": 7500,
            }
        )

        with patch("app.api.monitoring.redis_manager") as mock_redis:
            mock_redis.get_info = AsyncMock(return_value={"status": "healthy"})
            with patch("app.api.monitoring.embedding_service") as mock_embedding:
                mock_embedding.get_stats = MagicMock(return_value={"cache_hits": 7500})

                response = await async_client.get("/monitoring/cache/stats")
                data = response.json()
                assert any("High cache usage" in rec for rec in data["recommendations"])

    @patch("app.api.monitoring.metrics_collector")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_get_prometheus_metrics_error(self, mock_logger, mock_metrics, async_client):
        """Test Prometheus metrics error handling."""
        mock_metrics.get_metrics.side_effect = Exception("Metrics error")

        response = await async_client.get("/monitoring/metrics")

        assert response.status_code == 500
        assert "Error:" in response.text
        mock_logger.error.assert_called_once()

    @patch("app.api.monitoring.metrics_collector")
    @pytest.mark.asyncio
    async def test_get_metrics_json_success(self, mock_metrics, async_client):
        """Test successful JSON metrics retrieval."""
        mock_metrics.get_metrics_dict = MagicMock(
            return_value={
                "cache_hits": 100,
                "cache_misses": 20,
                "request_count": 120,
            }
        )

        response = await async_client.get("/monitoring/metrics/json")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert "timestamp" in data
        assert data["metrics"]["cache_hits"] == 100

    @patch("app.api.monitoring.metrics_collector")
    @patch("app.api.monitoring.logger")
    @pytest.mark.asyncio
    async def test_get_metrics_json_error(self, mock_logger, mock_metrics, async_client):
        """Test JSON metrics error handling."""
        mock_metrics.get_metrics_dict.side_effect = Exception("Metrics error")

        response = await async_client.get("/monitoring/metrics/json")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data
        mock_logger.error.assert_called_once()
