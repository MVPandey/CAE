"""Test main app lifespan events."""

from unittest.mock import AsyncMock, patch

import pytest


class TestAppLifespan:
    """Test app lifespan handling."""

    @pytest.mark.asyncio
    @patch("app.main.metrics_collector")
    @patch("app.main.redis_manager")
    @patch("app.main.db")
    async def test_lifespan_shutdown(self, mock_db, mock_redis, mock_metrics):
        """Test that shutdown cleans up resources properly."""
        from app.main import lifespan

        mock_app = AsyncMock()

        mock_redis.close = AsyncMock()
        mock_redis.initialize = AsyncMock()
        mock_db.create_db_and_tables = AsyncMock()
        mock_metrics.initialize = AsyncMock()

        async with lifespan(mock_app):
            pass

        mock_db.create_db_and_tables.assert_called_once()
        mock_redis.initialize.assert_called_once()
        mock_metrics.initialize.assert_called_once()

        mock_redis.close.assert_called_once()
