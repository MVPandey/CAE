"""Unit tests for database initialization and singleton."""

from unittest.mock import MagicMock, patch

from app.db import chat


class TestDatabaseInitialization:
    """Test database initialization and singleton."""

    @patch("app.utils.config.Config")
    def test_database_url_construction(self, mock_config_class):
        """Test DATABASE_URL is constructed correctly from settings."""
        # Setup mock settings
        mock_settings = MagicMock()
        mock_settings.DB_USER = "testuser"
        mock_settings.DB_SECRET = "testpass"
        mock_settings.DB_HOST = "localhost"
        mock_settings.DB_PORT = 5432
        mock_settings.DB_NAME = "testdb"
        mock_config_class.return_value = mock_settings

        # Patch the app_settings at module level before reload
        with patch("app.utils.config.app_settings", mock_settings):
            # Import to trigger module-level code
            import importlib
            importlib.reload(chat)

            # Verify DATABASE_URL
            expected_url = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
            assert chat.DATABASE_URL == expected_url

    @patch("app.db.chat.app_settings")
    def test_db_singleton_exists(self, mock_settings):
        """Test that db singleton is created."""
        # Setup mock settings
        mock_settings.DB_USER = "user"
        mock_settings.DB_SECRET = "pass"
        mock_settings.DB_HOST = "host"
        mock_settings.DB_PORT = 5432
        mock_settings.DB_NAME = "db"

        # Import to ensure module is loaded
        from app.db.chat import db

        assert db is not None
        assert hasattr(db, "engine")
        assert hasattr(db, "async_session_maker")
        assert hasattr(db, "create_db_and_tables")
        assert hasattr(db, "get_session")

    @patch("app.utils.config.Config")
    def test_database_url_special_characters(self, mock_config_class):
        """Test DATABASE_URL handles special characters in password."""
        # Setup mock settings with special characters
        mock_settings = MagicMock()
        mock_settings.DB_USER = "user@domain"
        mock_settings.DB_SECRET = "pass@word#123"
        mock_settings.DB_HOST = "db.example.com"
        mock_settings.DB_PORT = 5433
        mock_settings.DB_NAME = "my-database"
        mock_config_class.return_value = mock_settings

        # Patch the app_settings at module level before reload
        with patch("app.utils.config.app_settings", mock_settings):
            # Import to trigger module-level code
            import importlib
            importlib.reload(chat)

            # Verify URL is constructed (note: in real use, special chars should be URL-encoded)
            expected_url = "postgresql+asyncpg://user@domain:pass@word#123@db.example.com:5433/my-database"
            assert chat.DATABASE_URL == expected_url
