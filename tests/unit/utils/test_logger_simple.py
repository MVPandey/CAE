"""Simple tests for logger to increase coverage."""

import logging
from unittest.mock import MagicMock

from app.utils.logger import InterceptHandler


class TestInterceptHandler:
    """Test the InterceptHandler class."""

    def test_emit_uvicorn_debug(self):
        """Test that uvicorn debug logs are ignored."""
        handler = InterceptHandler()

        record = MagicMock()
        record.name = "uvicorn"
        record.levelno = logging.DEBUG

        result = handler.emit(record)
        assert result is None

    def test_emit_value_error_level(self):
        """Test handling ValueError when getting level name."""
        handler = InterceptHandler()

        record = MagicMock()
        record.name = "test"
        record.levelno = 25
        record.levelname = "CUSTOM"
        record.exc_info = None
        record.getMessage.return_value = "Test message"

        from app.utils.logger import logger as global_logger

        original_logger = global_logger._logger
        mock_logger = MagicMock()
        mock_logger.level.side_effect = ValueError("Unknown level")
        mock_logger.opt.return_value = mock_logger
        global_logger._logger = mock_logger

        try:
            handler.emit(record)

            mock_logger.opt.assert_called_once()

            mock_logger.log.assert_called_with(25, "Test message")
        finally:
            global_logger._logger = original_logger
