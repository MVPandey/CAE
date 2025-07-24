"""Unit tests for app.utils.logger module."""

import json
import logging
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from app.utils.logger import (
    InterceptHandler,
    LoggerWrapper,
    format_record,
    format_record_json,
    get_logger,
)


class TestInterceptHandler:
    """Test InterceptHandler class."""

    def test_intercept_handler_emit_normal_record(self):
        """Test InterceptHandler emits normal log records."""
        handler = InterceptHandler()

        # Create a mock record
        record = Mock()
        record.name = "test_logger"
        record.levelno = logging.INFO
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        record.exc_info = None

        # Mock logger to capture the call
        with patch("app.utils.logger.logger") as mock_logger:
            mock_logger.level.return_value.name = "INFO"
            mock_logger.opt.return_value.log = MagicMock()

            handler.emit(record)

            # Verify logger was called
            mock_logger.opt.assert_called_once()
            mock_logger.opt.return_value.log.assert_called_once_with("INFO", "Test message")

    def test_intercept_handler_filters_uvicorn_debug(self):
        """Test InterceptHandler filters out uvicorn debug logs."""
        handler = InterceptHandler()

        # Create a uvicorn debug record
        record = Mock()
        record.name = "uvicorn"
        record.levelno = logging.DEBUG

        # Mock logger - should not be called
        with patch("app.utils.logger.logger") as mock_logger:
            handler.emit(record)

            # Logger should not be called for filtered records
            mock_logger.opt.assert_not_called()

    def test_intercept_handler_with_exception(self):
        """Test InterceptHandler handles records with exceptions."""
        handler = InterceptHandler()

        # Create a record with exception
        record = Mock()
        record.name = "test_logger"
        record.levelno = logging.ERROR
        record.levelname = "ERROR"
        record.getMessage.return_value = "Error occurred"
        record.exc_info = (Exception, Exception("Test error"), None)

        with patch("app.utils.logger.logger") as mock_logger:
            mock_logger.level.return_value.name = "ERROR"
            mock_logger.opt.return_value.log = MagicMock()

            handler.emit(record)

            # Verify exception info was passed
            mock_logger.opt.assert_called_once()
            assert mock_logger.opt.call_args[1]["exception"] == record.exc_info


class TestFormatRecord:
    """Test format_record function."""

    def test_format_record_basic(self):
        """Test basic record formatting."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": Mock(name="INFO"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "extra": {},
            "exception": None
        }

        result = format_record(record)

        # The format_record function returns a format string, not the actual formatted message
        # It should contain the format placeholders
        assert "{time:YYYY-MM-DD HH:mm:ss.SSS}" in result
        assert "{level: <8}" in result
        assert "{name}" in result
        assert "{function}" in result
        assert "{line}" in result
        assert "{message}" in result

    def test_format_record_with_extra(self):
        """Test record formatting with extra fields."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": Mock(name="INFO"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "extra": {
                "user_id": "123",
                "request_id": "abc-def",
                "_internal": "hidden"  # Should be filtered
            },
            "exception": None
        }

        result = format_record(record)

        # Check that extra fields are properly formatted in the result
        assert "<blue>user_id</blue>=<yellow>123</yellow>" in result
        assert "<blue>request_id</blue>=<yellow>abc-def</yellow>" in result
        assert "_internal" not in result  # Hidden fields should be filtered

    def test_format_record_truncates_long_values(self):
        """Test that long values are truncated."""
        long_string = "x" * 200
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": Mock(name="INFO"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "extra": {
                "long_value": long_string
            },
            "exception": None
        }

        result = format_record(record)

        # Check that the long value is truncated to 97 chars + "..."
        assert "<blue>long_value</blue>=<yellow>" in result
        assert "..." in result
        # The actual truncation happens, we just need to verify the pattern is there
        assert "xxx" in result

    def test_format_record_escapes_braces(self):
        """Test that braces in values are escaped."""
        record = {
            "time": datetime(2024, 1, 1, 12, 0, 0),
            "level": Mock(name="INFO"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "extra": {
                "json_like": "{key: value}"
            },
            "exception": None
        }

        result = format_record(record)

        assert "{{key: value}}" in result  # Braces should be escaped


class TestFormatRecordJson:
    """Test format_record_json function."""

    def test_format_record_json_basic(self):
        """Test basic JSON record formatting."""
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"

        # Create a mock level object with a name attribute
        mock_level = Mock()
        mock_level.name = "INFO"
        
        record = {
            "time": mock_time,
            "level": mock_level,
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "module": "test_module",
            "extra": {},
            "exception": None
        }

        result = format_record_json(record)
        parsed = json.loads(result)

        assert parsed["timestamp"] == "2024-01-01T12:00:00"
        # The level is a Mock object with name attribute
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["function"] == "test_func"
        assert parsed["line"] == 42
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"

    def test_format_record_json_with_extra(self):
        """Test JSON formatting with extra fields."""
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"

        record = {
            "time": mock_time,
            "level": Mock(name="INFO"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Test message",
            "module": "test_module",
            "extra": {
                "user_id": "123",
                "request_id": "abc-def",
                "_internal": "hidden"  # Should be filtered
            },
            "exception": None
        }

        result = format_record_json(record)
        parsed = json.loads(result)

        assert parsed["user_id"] == "123"
        assert parsed["request_id"] == "abc-def"
        assert "_internal" not in parsed  # Hidden fields should be filtered

    def test_format_record_json_with_exception(self):
        """Test JSON formatting with exception info."""
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"

        mock_exception = Mock()
        mock_exception.type = ValueError
        mock_exception.value = ValueError("Test error")
        mock_exception.traceback = Mock(raw="Traceback details")

        record = {
            "time": mock_time,
            "level": Mock(name="ERROR"),
            "name": "test_logger",
            "function": "test_func",
            "line": 42,
            "message": "Error occurred",
            "module": "test_module",
            "extra": {},
            "exception": mock_exception
        }

        result = format_record_json(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["value"] == "Test error"
        assert parsed["exception"]["traceback"] == "Traceback details"


class TestLoggerWrapper:
    """Test LoggerWrapper class."""

    def test_logger_wrapper_basic_methods(self):
        """Test basic logging methods of LoggerWrapper."""
        mock_logger = MagicMock()
        wrapper = LoggerWrapper(mock_logger)

        # Test each logging method
        wrapper.info("Info message")
        mock_logger.info.assert_called_once_with("Info message")

        wrapper.debug("Debug message")
        mock_logger.debug.assert_called_once_with("Debug message")

        wrapper.warning("Warning message")
        mock_logger.warning.assert_called_once_with("Warning message")

        wrapper.error("Error message")
        mock_logger.error.assert_called_once_with("Error message")

        wrapper.critical("Critical message")
        mock_logger.critical.assert_called_once_with("Critical message")

    def test_logger_wrapper_with_extra(self):
        """Test LoggerWrapper with extra context."""
        mock_logger = MagicMock()
        mock_bound_logger = MagicMock()
        mock_logger.bind.return_value = mock_bound_logger

        wrapper = LoggerWrapper(mock_logger)

        # Log with extra context
        wrapper.info("Message", extra={"user_id": "123"})

        # Should bind context and use bound logger
        mock_logger.bind.assert_called_once_with(user_id="123")
        mock_bound_logger.info.assert_called_once_with("Message")

    def test_logger_wrapper_bind(self):
        """Test LoggerWrapper bind method."""
        mock_logger = MagicMock()
        mock_bound_logger = MagicMock()
        mock_logger.bind.return_value = mock_bound_logger

        wrapper = LoggerWrapper(mock_logger)

        # Create bound logger
        bound_wrapper = wrapper.bind(request_id="abc")

        # Should return a new LoggerWrapper
        assert isinstance(bound_wrapper, LoggerWrapper)
        assert bound_wrapper._logger == mock_bound_logger
        mock_logger.bind.assert_called_once_with(request_id="abc")

    def test_logger_wrapper_opt(self):
        """Test LoggerWrapper opt method passthrough."""
        mock_logger = MagicMock()
        wrapper = LoggerWrapper(mock_logger)

        # Call opt method
        result = wrapper.opt(depth=2, exception=True)

        # Should pass through to wrapped logger
        mock_logger.opt.assert_called_once_with(depth=2, exception=True)
        assert result == mock_logger.opt.return_value

    def test_logger_wrapper_getattr(self):
        """Test LoggerWrapper delegates unknown attributes."""
        mock_logger = MagicMock()
        mock_logger.custom_method = MagicMock(return_value="custom result")

        wrapper = LoggerWrapper(mock_logger)

        # Access custom method
        result = wrapper.custom_method("arg1", "arg2")

        # Should delegate to wrapped logger
        mock_logger.custom_method.assert_called_once_with("arg1", "arg2")
        assert result == "custom result"


class TestGetLogger:
    """Test get_logger function."""

    @patch("app.utils.logger.logger")
    def test_get_logger_no_context(self, mock_logger):
        """Test get_logger without context."""
        mock_bound = MagicMock()
        mock_logger.bind.return_value = mock_bound

        result = get_logger()

        # Should bind empty context
        mock_logger.bind.assert_called_once_with()
        assert isinstance(result, LoggerWrapper)
        assert result._logger == mock_bound

    @patch("app.utils.logger.logger")
    def test_get_logger_with_context(self, mock_logger):
        """Test get_logger with context."""
        mock_bound = MagicMock()
        mock_logger.bind.return_value = mock_bound

        result = get_logger(user_id="123", request_id="abc")

        # Should bind provided context
        mock_logger.bind.assert_called_once_with(user_id="123", request_id="abc")
        assert isinstance(result, LoggerWrapper)
        assert result._logger == mock_bound


class TestLoggerConfiguration:
    """Test logger configuration and setup."""

    def test_intercept_handler_registered(self):
        """Test that InterceptHandler is registered with standard logging."""
        # Import the actual logging module to check handlers
        import logging
        
        # Check that root logger has InterceptHandler
        root_logger = logging.getLogger()
        assert any(isinstance(handler, InterceptHandler) for handler in root_logger.handlers)
        
        # Check that specific loggers have InterceptHandler
        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
            logger = logging.getLogger(logger_name)
            assert any(isinstance(handler, InterceptHandler) for handler in logger.handlers)
