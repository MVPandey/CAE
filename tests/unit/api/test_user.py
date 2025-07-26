"""Unit tests for the user API endpoints."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status

from app.schema.llm.chat import Chat
from app.schema.user import User


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    return User(id=uuid4(), name="Test User", created_at="2024-01-01T00:00:00")


@pytest.fixture
def mock_users(mock_user):
    """Create a list of mock users."""
    return [mock_user, User(id=uuid4(), name="Another User", created_at="2024-01-02T00:00:00")]


@pytest.fixture
def mock_chats():
    """Create a list of mock chats."""
    return [
        Chat(id=uuid4(), user_id=uuid4(), created_at="2024-01-01T00:00:00"),
        Chat(id=uuid4(), user_id=uuid4(), created_at="2024-01-02T00:00:00"),
    ]


class TestCreateUser:
    """Test cases for POST /users/ endpoint."""

    @pytest.mark.asyncio
    async def test_create_user_success(self, async_client, mock_user):
        """Test successful user creation."""
        with patch("app.api.user.db.create_user", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_user

            response = await async_client.post("/users/", json={"name": "Test User"})

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["name"] == "Test User"
            assert "id" in data
            assert "created_at" in data
            mock_create.assert_called_once_with(name="Test User")

    @pytest.mark.asyncio
    async def test_create_user_empty_name(self, async_client):
        """Test user creation with empty name."""
        response = await async_client.post("/users/", json={"name": ""})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_create_user_missing_name(self, async_client):
        """Test user creation without name field."""
        response = await async_client.post("/users/", json={})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_create_user_long_name(self, async_client):
        """Test user creation with name exceeding max length."""
        response = await async_client.post(
            "/users/",
            json={"name": "a" * 256},  # Exceeds max_length=255
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_create_user_database_error(self, async_client):
        """Test user creation when database error occurs."""
        with patch("app.api.user.db.create_user", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Database connection failed")

            response = await async_client.post("/users/", json={"name": "Test User"})

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Database connection failed" in response.json()["detail"]


class TestListUsers:
    """Test cases for GET /users/ endpoint."""

    @pytest.mark.asyncio
    async def test_list_users_success(self, async_client, mock_users):
        """Test successful listing of all users."""
        with patch("app.api.user.db.list_users", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_users

            response = await async_client.get("/users/")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["name"] == "Test User"
            assert data[1]["name"] == "Another User"
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_users_empty(self, async_client):
        """Test listing users when no users exist."""
        with patch("app.api.user.db.list_users", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            response = await async_client.get("/users/")

            assert response.status_code == status.HTTP_200_OK
            assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_users_database_error(self, async_client):
        """Test listing users when database error occurs."""
        with patch("app.api.user.db.list_users", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            response = await async_client.get("/users/")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Database error" in response.json()["detail"]


class TestGetUser:
    """Test cases for GET /users/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_user_success(self, async_client, mock_user):
        """Test successful retrieval of a specific user."""
        user_id = mock_user.id
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_user

            response = await async_client.get(f"/users/{user_id}")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["name"] == "Test User"
            assert data["id"] == str(user_id)
            mock_get.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, async_client):
        """Test getting a non-existent user."""
        user_id = uuid4()
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await async_client.get(f"/users/{user_id}")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json()["detail"] == "User not found"

    @pytest.mark.asyncio
    async def test_get_user_invalid_uuid(self, async_client):
        """Test getting user with invalid UUID format."""
        response = await async_client.get("/users/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_user_database_error(self, async_client):
        """Test getting user when database error occurs."""
        user_id = uuid4()
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Connection timeout")

            response = await async_client.get(f"/users/{user_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Connection timeout" in response.json()["detail"]


class TestDeleteUser:
    """Test cases for DELETE /users/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_user_success(self, async_client):
        """Test successful user deletion."""
        user_id = uuid4()
        with patch("app.api.user.db.delete_user", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True

            response = await async_client.delete(f"/users/{user_id}")

            assert response.status_code == status.HTTP_204_NO_CONTENT
            assert response.content == b""
            mock_delete.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, async_client):
        """Test deleting a non-existent user."""
        user_id = uuid4()
        with patch("app.api.user.db.delete_user", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = False

            response = await async_client.delete(f"/users/{user_id}")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json()["detail"] == "User not found"

    @pytest.mark.asyncio
    async def test_delete_user_invalid_uuid(self, async_client):
        """Test deleting user with invalid UUID format."""
        response = await async_client.delete("/users/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_delete_user_database_error(self, async_client):
        """Test deleting user when database error occurs."""
        user_id = uuid4()
        with patch("app.api.user.db.delete_user", new_callable=AsyncMock) as mock_delete:
            mock_delete.side_effect = Exception("Foreign key constraint violation")

            response = await async_client.delete(f"/users/{user_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Foreign key constraint violation" in response.json()["detail"]


class TestGetUserChats:
    """Test cases for GET /users/{user_id}/chats endpoint."""

    @pytest.mark.asyncio
    async def test_get_user_chats_success(self, async_client, mock_user, mock_chats):
        """Test successful retrieval of user's chats."""
        user_id = mock_user.id
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get_user:
            with patch("app.api.user.db.get_user_chats", new_callable=AsyncMock) as mock_get_chats:
                mock_get_user.return_value = mock_user
                mock_get_chats.return_value = mock_chats

                response = await async_client.get(f"/users/{user_id}/chats")

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert len(data) == 2
                assert all("id" in chat for chat in data)
                assert all("user_id" in chat for chat in data)
                mock_get_user.assert_called_once_with(user_id)
                mock_get_chats.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_get_user_chats_user_not_found(self, async_client):
        """Test getting chats for non-existent user."""
        user_id = uuid4()
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = None

            response = await async_client.get(f"/users/{user_id}/chats")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json()["detail"] == "User not found"

    @pytest.mark.asyncio
    async def test_get_user_chats_empty(self, async_client, mock_user):
        """Test getting chats when user has no chats."""
        user_id = mock_user.id
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get_user:
            with patch("app.api.user.db.get_user_chats", new_callable=AsyncMock) as mock_get_chats:
                mock_get_user.return_value = mock_user
                mock_get_chats.return_value = []

                response = await async_client.get(f"/users/{user_id}/chats")

                assert response.status_code == status.HTTP_200_OK
                assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_user_chats_invalid_uuid(self, async_client):
        """Test getting user chats with invalid UUID format."""
        response = await async_client.get("/users/invalid-uuid-format/chats")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_user_chats_database_error(self, async_client, mock_user):
        """Test getting user chats when database error occurs."""
        user_id = mock_user.id
        with patch("app.api.user.db.get_user", new_callable=AsyncMock) as mock_get_user:
            with patch("app.api.user.db.get_user_chats", new_callable=AsyncMock) as mock_get_chats:
                mock_get_user.return_value = mock_user
                mock_get_chats.side_effect = Exception("Query timeout")

                response = await async_client.get(f"/users/{user_id}/chats")

                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                assert "Query timeout" in response.json()["detail"]
