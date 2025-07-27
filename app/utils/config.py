from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        str_strip_whitespace=True,
        strict=False,
    )

    LLM_API_KEY: str
    LLM_API_BASE_URL: str
    LLM_MODEL_NAME: str
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL_BASE_URL: str
    EMBEDDING_MODEL_NAME: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_SECRET: str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0
    REDIS_POOL_SIZE: int = 20
    REDIS_MAX_CONNECTIONS: int = 50
    CACHE_TTL_SECONDS: int = 3600  # 1 hour default
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    CACHE_MAX_ENTRIES: int = 10000
    LOG_LEVEL: str | None = "INFO"
    LLM_TIMEOUT_SECONDS: int = 600  # Default 10 minutes, can be overridden by env var


app_settings = Config()
