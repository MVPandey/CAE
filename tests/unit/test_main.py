"""Unit tests for app.main module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.main import (
    app,
    general_exception_handler,
    health_check,
    http_exception_handler,
    lifespan,
    log_requests,
    validation_exception_handler,
)


class TestLifespan:
    """Test lifespan context manager."""

    @pytest.mark.asyncio
    @patch("app.main.db")
    @patch("app.main.logger")
    async def test_lifespan_startup_shutdown(self, mock_logger, mock_db):
        """Test lifespan handles startup and shutdown."""
        mock_db.create_db_and_tables = AsyncMock()

        # Test lifespan
        async with lifespan(app) as _:
            # Startup logs
            assert mock_logger.info.call_count >= 2
            startup_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert "Starting up..." in startup_calls
            assert "Database tables created or already exist." in startup_calls

            # Database initialized
            mock_db.create_db_and_tables.assert_called_once()

        # Shutdown logs
        shutdown_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Shutting down..." in shutdown_calls

    @pytest.mark.asyncio
    @patch("app.main.db")
    @patch("app.main.logger")
    async def test_lifespan_database_error(self, mock_logger, mock_db):
        """Test lifespan handles database initialization errors."""
        mock_db.create_db_and_tables = AsyncMock(side_effect=Exception("DB Error"))

        with pytest.raises(Exception) as exc_info:
            async with lifespan(app):
                pass

        assert str(exc_info.value) == "DB Error"
        mock_logger.info.assert_called_with("Starting up...")


class TestApp:
    """Test FastAPI app configuration."""

    def test_app_configuration(self):
        """Test app is configured correctly."""
        assert app.title == "CAE API"
        assert app.description == "API for Conversational Analysis Engine"
        assert app.version == "0.0.1"

        # Check CORS middleware is added
        middleware_classes = [m.cls for m in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_classes

    def test_routers_included(self):
        """Test all routers are included."""
        routes = [route.path for route in app.routes]

        # Check health endpoint
        assert "/health" in routes

        # Check user endpoints (from router)
        assert any("/users" in route for route in routes)

        # Check chat endpoints (from router)
        assert any("/chats" in route for route in routes)

        # Check analysis endpoints (from router)
        assert any("/analysis" in route for route in routes)


class TestMiddleware:
    """Test custom middleware."""

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_log_requests_middleware(self, mock_logger):
        """Test request logging middleware."""
        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.query_params = {"param": "value"}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200

        # Mock call_next
        async def mock_call_next(request):
            return mock_response

        # Test middleware
        with patch("app.main.time.time") as mock_time:
            mock_time.side_effect = [1.0, 2.5]  # Start and end time

            result = await log_requests(mock_request, mock_call_next)

        # Verify logging
        assert mock_logger.info.call_count == 2

        # Check request log
        first_call = mock_logger.info.call_args_list[0]
        assert "Incoming request: GET /test" in first_call[0][0]
        assert first_call[1]["extra"]["method"] == "GET"
        assert first_call[1]["extra"]["path"] == "/test"
        assert first_call[1]["extra"]["query_params"] == {"param": "value"}
        assert first_call[1]["extra"]["body"] == '{"test": "data"}'

        # Check response log
        second_call = mock_logger.info.call_args_list[1]
        assert "Request completed: GET /test" in second_call[0][0]
        assert second_call[1]["extra"]["status_code"] == 200
        assert second_call[1]["extra"]["process_time"] == "1.500s"

        assert result == mock_response

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_log_requests_no_body(self, mock_logger):
        """Test request logging with no body."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=b"")

        mock_response = Mock()
        mock_response.status_code = 200

        async def mock_call_next(request):
            return mock_response

        await log_requests(mock_request, mock_call_next)

        # Check body is None when empty
        first_call = mock_logger.info.call_args_list[0]
        assert first_call[1]["extra"]["body"] is None


class TestExceptionHandlers:
    """Test exception handlers."""

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_validation_exception_handler(self, mock_logger):
        """Test request validation error handler."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "POST"

        exc = RequestValidationError(
            [{"loc": ("body", "field"), "msg": "field required", "type": "value_error"}],
            body={"incomplete": "data"}
        )

        response = await validation_exception_handler(mock_request, exc)

        # Check logging
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Validation error for POST /test" in log_message

        # Check response
        assert response.status_code == 422
        content = response.body.decode()
        assert "field required" in content
        assert "Request validation failed" in content

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_http_exception_handler(self, mock_logger):
        """Test HTTP exception handler."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "GET"

        exc = StarletteHTTPException(status_code=404, detail="Not found")

        response = await http_exception_handler(mock_request, exc)

        # Check logging
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "HTTP exception for GET /test | Not found" in log_message
        assert mock_logger.error.call_args[1]["extra"]["status_code"] == 404

        # Check response
        assert response.status_code == 404
        assert b"Not found" in response.body

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_general_exception_handler(self, mock_logger):
        """Test general exception handler."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "POST"

        exc = ValueError("Something went wrong")

        response = await general_exception_handler(mock_request, exc)

        # Check logging
        mock_logger.exception.assert_called_once()
        log_message = mock_logger.exception.call_args[0][0]
        assert "Unhandled exception for POST /test" in log_message
        assert mock_logger.exception.call_args[1]["extra"]["exception_type"] == "ValueError"
        assert mock_logger.exception.call_args[1]["extra"]["exception_message"] == "Something went wrong"

        # Check response
        assert response.status_code == 500
        assert b"Internal server error" in response.body


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns correct response."""
        result = await health_check()
        assert result == {"status": "healthy"}

    @pytest.mark.asyncio
    async def test_health_check_via_client(self, async_client):
        """Test health check endpoint via test client."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestMainModule:
    """Test main module execution."""

    @patch("app.main.run_uvicorn")
    def test_main_execution(self, mock_run_uvicorn):
        """Test main module runs uvicorn when executed directly."""
        # Simulate running as main module
        with patch("app.main.__name__", "__main__"):
            import app.main

        # Verify run_uvicorn was called
        mock_run_uvicorn.assert_called_once()

    @patch("app.main.uvicorn")
    def test_run_uvicorn_function(self, mock_uvicorn):
        """Test run_uvicorn function calls uvicorn.run with correct parameters."""
        from app.main import run_uvicorn, app
        
        run_uvicorn()

        # Verify uvicorn.run was called with correct parameters
        mock_uvicorn.run.assert_called_once_with(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
        )
