"""Simple tests for monitoring endpoints."""

from unittest.mock import MagicMock, patch

import pytest




class TestMonitoringEndpoints:
    """Test monitoring endpoints with proper async handling."""

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_metrics_prometheus_format(self, mock_collector, async_client):
        """Test Prometheus metrics endpoint."""
        mock_collector.get_metrics.return_value = b"# TYPE test_metric counter\ntest_metric 42\n"

        response = await async_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert b"test_metric" in response.content

    @patch("app.utils.metrics.metrics_collector")
    @pytest.mark.asyncio
    async def test_metrics_json_format(self, mock_collector, async_client):
        """Test JSON metrics endpoint."""
        mock_collector.get_metrics_dict.return_value = {
            "test_counter": 42.0,
            "test_gauge": 3.14,
        }

        response = await async_client.get("/metrics/json")

        assert response.status_code == 200
        data = response.json()
        assert data["test_counter"] == 42.0
        assert data["test_gauge"] == 3.14

    @patch("app.services.cache.semantic_cache.semantic_cache")
    @patch("app.services.cache.redis_manager.redis_manager")
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_redis, mock_cache, async_client):
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

    @patch("app.services.cache.semantic_cache.semantic_cache")
    @patch("app.services.cache.redis_manager.redis_manager")
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_redis, mock_cache, async_client):
        """Test health check when a service is unhealthy."""
        mock_redis.is_healthy = False

        async def mock_health_check():
            return True

        mock_cache.health_check = mock_health_check

        response = await async_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["redis"]["status"] == "unhealthy"

    @patch("app.services.cache.semantic_cache.semantic_cache")
    @patch("app.services.cache.redis_manager.redis_manager")
    @pytest.mark.asyncio
    async def test_health_detailed(self, mock_redis, mock_cache, async_client):
        """Test detailed health check endpoint."""
        mock_redis.is_healthy = True
        mock_redis.get_connection_info = MagicMock(
            return_value={
                "host": "localhost",
                "port": 6379,
                "is_healthy": True,
            }
        )

        async def mock_health_check():
            return True

        mock_cache.health_check = mock_health_check

        async def mock_get_stats():
            return {
                "total_entries": 100,
                "memory_usage": 1024,
            }

        mock_cache.get_stats = mock_get_stats

        response = await async_client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert "connection_info" in data["services"]["redis"]
        assert "stats" in data["services"]["cache"]
