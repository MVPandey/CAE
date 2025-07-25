"""Unit tests for user schema models."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schema.user import User, UserCreate


class TestUser:
    """Tests for User model."""

    def test_user_creation_with_defaults(self):
        """Test creating a User with default values."""
        user = User(name="John Doe")

        assert isinstance(user.id, UUID)
        assert user.name == "John Doe"
        assert isinstance(user.created_at, datetime)

        time_diff = datetime.now(UTC) - user.created_at
        assert time_diff.total_seconds() < 60

    def test_user_creation_with_explicit_values(self):
        """Test creating a User with explicit values."""
        user_id = uuid4()
        created_at = datetime(2024, 1, 1, 12, 0, 0)

        user = User(id=user_id, name="Jane Smith", created_at=created_at)

        assert user.id == user_id
        assert user.name == "Jane Smith"
        assert user.created_at == created_at

    def test_user_from_attributes(self):
        """Test that User can be created from ORM attributes."""

        class MockORMUser:
            id = uuid4()
            name = "Test User"
            created_at = datetime.now(UTC)

        orm_user = MockORMUser()
        user = User.model_validate(orm_user, from_attributes=True)

        assert user.id == orm_user.id
        assert user.name == orm_user.name
        assert user.created_at == orm_user.created_at

    def test_user_name_constraints(self):
        """Test name field constraints."""
        user = User(name="A")
        assert user.name == "A"

        long_name = "A" * 255
        user = User(name=long_name)
        assert user.name == long_name
        assert len(user.name) == 255

    def test_user_name_too_short(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            User(name="")

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" and "at least 1 character" in str(error) for error in errors)

    def test_user_name_too_long(self):
        """Test that name exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            User(name="A" * 256)

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" and "at most 255 characters" in str(error) for error in errors)

    def test_user_missing_name(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            User()

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" for error in errors)

    def test_user_invalid_id(self):
        """Test that invalid id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            User(id="not-a-uuid", name="Test")

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "id" for error in errors)

    def test_user_special_characters_in_name(self):
        """Test that special characters are allowed in name."""
        special_names = ["John O'Brien", "María García", "李明", "Müller, Hans", "user@example.com", "User-123"]

        for name in special_names:
            user = User(name=name)
            assert user.name == name


class TestUserCreate:
    """Tests for UserCreate model."""

    def test_user_create_valid(self):
        """Test creating a valid UserCreate request."""
        user_create = UserCreate(name="New User")
        assert user_create.name == "New User"

    def test_user_create_name_constraints(self):
        """Test UserCreate name field constraints."""
        user_create = UserCreate(name="A")
        assert user_create.name == "A"

        long_name = "B" * 255
        user_create = UserCreate(name=long_name)
        assert user_create.name == long_name

    def test_user_create_name_too_short(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(name="")

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" and "at least 1 character" in str(error) for error in errors)

    def test_user_create_name_too_long(self):
        """Test that name exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(name="X" * 256)

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" and "at most 255 characters" in str(error) for error in errors)

    def test_user_create_missing_name(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UserCreate()

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "name" for error in errors)

    def test_user_create_whitespace_handling(self):
        """Test how UserCreate handles whitespace in names."""
        test_cases = [
            ("  John Doe  ", "  John Doe  "),
            ("John  Doe", "John  Doe"),
            (" ", " "),
        ]

        for input_name, expected_name in test_cases:
            user_create = UserCreate(name=input_name)
            assert user_create.name == expected_name

    def test_user_create_to_user_conversion(self):
        """Test that UserCreate can be used to create a User."""
        user_create = UserCreate(name="Test User")

        user = User(name=user_create.name)

        assert user.name == user_create.name
        assert isinstance(user.id, UUID)
        assert isinstance(user.created_at, datetime)
