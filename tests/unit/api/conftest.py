"""Fixtures for API unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.chat_service import ChatService
from app.services.conversation_analysis_service import ConversationAnalysisService


@pytest.fixture
def test_client():
    """Create a synchronous test client."""
    return TestClient(app)


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    service = MagicMock(spec=ChatService)
    service.process_message = AsyncMock()
    return service


@pytest.fixture
def mock_conversation_analysis_service():
    """Create a mock ConversationAnalysisService."""
    service = MagicMock(spec=ConversationAnalysisService)
    service.analyze_conversation = AsyncMock()
    return service


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()
