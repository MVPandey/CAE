"""Unit tests for the chat API endpoints - simplified version."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status

from app.main import app
from app.schema.llm.chat import ChatMessage, ChatRole
from app.services.chat_service import ChatService


@pytest.fixture
def mock_chat_messages():
    """Create mock chat messages."""
    return [
        ChatMessage(
            id=uuid4(),
            chat_id=uuid4(),
            role=ChatRole.USER,
            content="Hello, how are you?",
            created_at="2024-01-01T00:00:00"
        ),
        ChatMessage(
            id=uuid4(),
            chat_id=uuid4(),
            role=ChatRole.ASSISTANT,
            content="I'm doing well, thank you! How can I help you today?",
            created_at="2024-01-01T00:00:01"
        )
    ]


class TestChatEndpoints:
    """Test cases for chat endpoints."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, async_client, mock_chat_messages):
        """Test successful message sending with dependency override."""
        # Create a mock service
        mock_service = AsyncMock(spec=ChatService)
        mock_service.process_message = AsyncMock(return_value=mock_chat_messages)

        # Override the dependency
        app.dependency_overrides[ChatService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/chats/",
                json={
                    "user_id": str(uuid4()),
                    "message": "Hello, how are you?"
                }
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "Hello, how are you?"
            assert data[1]["role"] == "assistant"

            # Verify the service was called
            mock_service.process_message.assert_called_once()
            call_args = mock_service.process_message.call_args
            # Check that the user_id was passed correctly
            assert "user_id" in call_args.kwargs
            assert call_args.kwargs["user_message"] == "Hello, how are you?"
            assert call_args.kwargs["chat_id"] is None
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_send_message_with_chat_id(self, async_client, mock_chat_messages):
        """Test sending message to existing chat."""
        chat_id = str(uuid4())
        user_id = str(uuid4())

        # Create a mock service
        mock_service = AsyncMock(spec=ChatService)
        mock_service.process_message = AsyncMock(return_value=mock_chat_messages)

        # Override the dependency
        app.dependency_overrides[ChatService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/chats/",
                json={
                    "user_id": user_id,
                    "message": "What's the weather like?",
                    "chat_id": chat_id
                }
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2

            # Verify the service was called with correct parameters
            mock_service.process_message.assert_called_once()
            call_args = mock_service.process_message.call_args
            assert call_args.kwargs["chat_id"] == chat_id
            assert call_args.kwargs["user_id"] == user_id
            assert call_args.kwargs["user_message"] == "What's the weather like?"
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_send_message_service_error(self, async_client):
        """Test handling of service errors."""
        # Create a mock service that raises an error
        mock_service = AsyncMock(spec=ChatService)
        mock_service.process_message = AsyncMock(side_effect=Exception("LLM API error"))

        # Override the dependency
        app.dependency_overrides[ChatService] = lambda: mock_service

        try:
            response = await async_client.post(
                "/chats/",
                json={
                    "user_id": str(uuid4()),
                    "message": "Hello"
                }
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "LLM API error" in response.json()["detail"]
        finally:
            # Clean up
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_send_message_validation_errors(self, async_client):
        """Test request validation."""
        # Missing user_id
        response = await async_client.post(
            "/chats/",
            json={"message": "Hello"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Missing message
        response = await async_client.post(
            "/chats/",
            json={"user_id": str(uuid4())}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Empty request body
        response = await async_client.post(
            "/chats/",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_chat_history_success(self, async_client, mock_chat_messages):
        """Test successful chat history retrieval."""
        chat_id = str(uuid4())

        with patch("app.api.chat.db.get_chat_history", new_callable=AsyncMock) as mock_get_history:
            mock_get_history.return_value = mock_chat_messages

            response = await async_client.get(f"/chats/{chat_id}")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[1]["role"] == "assistant"
            mock_get_history.assert_called_once_with(chat_id)

    @pytest.mark.asyncio
    async def test_get_chat_history_not_found(self, async_client):
        """Test getting history for non-existent chat."""
        chat_id = str(uuid4())

        with patch("app.api.chat.db.get_chat_history", new_callable=AsyncMock) as mock_get_history:
            mock_get_history.return_value = []

            response = await async_client.get(f"/chats/{chat_id}")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json()["detail"] == "Chat not found"

    @pytest.mark.asyncio
    async def test_delete_chat_success(self, async_client):
        """Test successful chat deletion."""
        chat_id = str(uuid4())

        with patch("app.api.chat.db.delete_chat_session", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = None

            response = await async_client.delete(f"/chats/{chat_id}")

            assert response.status_code == status.HTTP_204_NO_CONTENT
            assert response.content == b''
            mock_delete.assert_called_once_with(chat_id)
