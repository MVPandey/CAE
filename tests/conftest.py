import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["DISABLE_PROMETHEUS_METRICS"] = "false"

TEST_CONSTANTS_CLASS_NAME = "TestConstants"


def should_skip_mock(request):
    """Helper function to check if mocking should be skipped for a test."""
    return request.node.parent and request.node.parent.name == TEST_CONSTANTS_CLASS_NAME


@pytest.fixture(scope="session")
def event_loop_policy():
    """Create a custom event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    yield


@pytest.fixture
async def async_client():
    """Create an async test client for FastAPI."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def mock_retry_delays(request):
    """Automatically mock retry delays for all tests to speed them up."""
    if should_skip_mock(request):
        yield
        return

    with (
        patch("app.utils.constants.RETRY_MIN_WAIT", 0.001),
        patch("app.utils.constants.RETRY_MAX_WAIT", 0.01),
        patch("app.utils.constants.RETRY_MULTIPLIER", 0.001),
        patch("app.utils.constants.DEFAULT_LLM_TIMEOUT", 1),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_tenacity_retry():
    """Mock tenacity retry decorators to speed up tests."""
    from tenacity import retry as original_retry

    def fast_retry(*args, **kwargs):
        kwargs["wait"] = lambda retry_state: 0.001
        return original_retry(*args, **kwargs)

    with patch("tenacity.retry", side_effect=fast_retry):
        yield


@pytest.fixture(autouse=True)
def mock_asyncio_sleep(request):
    """Mock asyncio.sleep to speed up tests."""
    if should_skip_mock(request):
        yield
        return

    import asyncio

    original_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await original_sleep(min(seconds * 0.001, 0.001))

    with patch("asyncio.sleep", side_effect=fast_sleep):
        yield
