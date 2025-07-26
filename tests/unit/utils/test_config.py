"""Unit tests for app.utils.config module."""

import os

import pytest
from pydantic import ValidationError

from app.utils.config import Config


class TestConfig:
    """Test Config class."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        env_vars = {
            "LLM_API_KEY": "test-llm-key",
            "LLM_API_BASE_URL": "https://api.llm.test",
            "LLM_MODEL_NAME": "test-model",
            "EMBEDDING_MODEL_API_KEY": "test-embed-key",
            "EMBEDDING_MODEL_BASE_URL": "https://api.embed.test",
            "EMBEDDING_MODEL_NAME": "test-embed-model",
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "testdb",
            "DB_USER": "testuser",
            "DB_SECRET": "testpass",
            "LOG_LEVEL": "DEBUG",
            "LLM_TIMEOUT_SECONDS": "300"
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        return env_vars

    def test_config_loads_from_env(self, mock_env_vars):
        """Test Config loads values from environment variables."""
        config = Config()

        assert config.LLM_API_KEY == "test-llm-key"
        assert config.LLM_API_BASE_URL == "https://api.llm.test"
        assert config.LLM_MODEL_NAME == "test-model"
        assert config.EMBEDDING_MODEL_API_KEY == "test-embed-key"
        assert config.EMBEDDING_MODEL_BASE_URL == "https://api.embed.test"
        assert config.EMBEDDING_MODEL_NAME == "test-embed-model"
        assert config.DB_HOST == "localhost"
        assert config.DB_PORT == 5432
        assert config.DB_NAME == "testdb"
        assert config.DB_USER == "testuser"
        assert config.DB_SECRET == "testpass"
        assert config.LOG_LEVEL == "DEBUG"
        assert config.LLM_TIMEOUT_SECONDS == 300

    def test_config_default_values(self, monkeypatch):
        """Test Config default values."""
        env_vars = {
            "LLM_API_KEY": "key",
            "LLM_API_BASE_URL": "url",
            "LLM_MODEL_NAME": "model",
            "EMBEDDING_MODEL_API_KEY": "key",
            "EMBEDDING_MODEL_BASE_URL": "url",
            "EMBEDDING_MODEL_NAME": "model",
            "DB_HOST": "host",
            "DB_PORT": "5432",
            "DB_NAME": "db",
            "DB_USER": "user",
            "DB_SECRET": "secret"
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = Config()

        assert config.LOG_LEVEL == "INFO"  # Default value
        assert config.LLM_TIMEOUT_SECONDS == 600  # Default value

    def test_config_strip_whitespace(self, monkeypatch):
        """Test that Config strips whitespace from string values."""
        monkeypatch.setenv("LLM_API_KEY", "  test-key  ")
        monkeypatch.setenv("LLM_API_BASE_URL", "\nurl\n")
        monkeypatch.setenv("LLM_MODEL_NAME", "model ")
        monkeypatch.setenv("EMBEDDING_MODEL_API_KEY", " key")
        monkeypatch.setenv("EMBEDDING_MODEL_BASE_URL", "url")
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "model")
        monkeypatch.setenv("DB_HOST", "  host  ")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_SECRET", "secret")

        config = Config()

        assert config.LLM_API_KEY == "test-key"
        assert config.LLM_API_BASE_URL == "url"
        assert config.LLM_MODEL_NAME == "model"
        assert config.DB_HOST == "host"

    def test_config_type_conversion(self, mock_env_vars):
        """Test that Config converts types correctly."""
        config = Config()

        assert isinstance(config.DB_PORT, int)
        assert config.DB_PORT == 5432

        assert isinstance(config.LLM_TIMEOUT_SECONDS, int)
        assert config.LLM_TIMEOUT_SECONDS == 300

    def test_config_missing_required_fields(self, monkeypatch, tmp_path):
        """Test Config raises error when required fields are missing."""
        monkeypatch.chdir(tmp_path)

        for key in os.environ.copy():
            if key.startswith(("LLM_", "EMBEDDING_", "DB_")):
                monkeypatch.delenv(key, raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Config()

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        expected_fields = {
            "LLM_API_KEY", "LLM_API_BASE_URL", "LLM_MODEL_NAME",
            "EMBEDDING_MODEL_API_KEY", "EMBEDDING_MODEL_BASE_URL", "EMBEDDING_MODEL_NAME",
            "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_SECRET"
        }
        assert expected_fields.issubset(error_fields)

    def test_config_invalid_port(self, monkeypatch):
        """Test Config with invalid port value."""
        env_vars = {
            "LLM_API_KEY": "key",
            "LLM_API_BASE_URL": "url",
            "LLM_MODEL_NAME": "model",
            "EMBEDDING_MODEL_API_KEY": "key",
            "EMBEDDING_MODEL_BASE_URL": "url",
            "EMBEDDING_MODEL_NAME": "model",
            "DB_HOST": "host",
            "DB_PORT": "not-a-number",  # Invalid
            "DB_NAME": "db",
            "DB_USER": "user",
            "DB_SECRET": "secret"
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        with pytest.raises(ValidationError) as exc_info:
            Config()

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "DB_PORT" for error in errors)

    def test_config_env_file_loading(self, tmp_path, monkeypatch):
        """Test Config can load from .env file."""
        env_file = tmp_path / ".env"
        env_content = """
LLM_API_KEY=file-llm-key
LLM_API_BASE_URL=https://file.llm.test
LLM_MODEL_NAME=file-model
EMBEDDING_MODEL_API_KEY=file-embed-key
EMBEDDING_MODEL_BASE_URL=https://file.embed.test
EMBEDDING_MODEL_NAME=file-embed-model
DB_HOST=file-host
DB_PORT=6432
DB_NAME=filedb
DB_USER=fileuser
DB_SECRET=filepass
LOG_LEVEL=WARNING
"""
        env_file.write_text(env_content)

        monkeypatch.chdir(tmp_path)

        for key in ["LLM_API_KEY", "LLM_API_BASE_URL", "LLM_MODEL_NAME",
                   "EMBEDDING_MODEL_API_KEY", "EMBEDDING_MODEL_BASE_URL", "EMBEDDING_MODEL_NAME",
                   "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_SECRET", "LOG_LEVEL"]:
            monkeypatch.delenv(key, raising=False)

        config = Config()

        assert config.LLM_API_KEY == "file-llm-key"
        assert config.DB_PORT == 6432
        assert config.LOG_LEVEL == "WARNING"

    def test_config_env_override_file(self, tmp_path, monkeypatch):
        """Test that environment variables override .env file values."""
        env_file = tmp_path / ".env"
        env_content = """
LLM_API_KEY=file-key
DB_PORT=6432
"""
        env_file.write_text(env_content)

        monkeypatch.chdir(tmp_path)

        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("DB_PORT", "7432")

        monkeypatch.setenv("LLM_API_BASE_URL", "url")
        monkeypatch.setenv("LLM_MODEL_NAME", "model")
        monkeypatch.setenv("EMBEDDING_MODEL_API_KEY", "key")
        monkeypatch.setenv("EMBEDDING_MODEL_BASE_URL", "url")
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "model")
        monkeypatch.setenv("DB_HOST", "host")
        monkeypatch.setenv("DB_NAME", "db")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_SECRET", "secret")

        config = Config()

        assert config.LLM_API_KEY == "env-key"
        assert config.DB_PORT == 7432

    def test_config_model_config_settings(self):
        """Test Config model configuration settings."""
        assert Config.model_config["env_file"] == ".env"
        assert Config.model_config["env_file_encoding"] == "utf-8"
        assert Config.model_config["str_strip_whitespace"] is True
        assert Config.model_config["strict"] is False


class TestAppSettings:
    """Test the app_settings singleton."""

    def test_app_settings_singleton(self):
        """Test that app_settings is properly initialized."""
        from app.utils.config import app_settings

        assert app_settings.LLM_API_KEY is not None
        assert isinstance(app_settings.LLM_API_KEY, str)
        assert len(app_settings.LLM_API_KEY) > 0

        assert isinstance(app_settings.DB_PORT, int)
        assert app_settings.DB_PORT > 0

        assert app_settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
