from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        str_strip_whitespace=True,
        strict=False,
        extra="ignore",
    )

    # ================================================================
    # REQUIRED: Core LLM configuration (only this is mandatory)
    # ================================================================
    LLM_API_KEY: str = Field(..., description="LLM API key (required)")

    # ================================================================
    # LLM Configuration with smart defaults
    # ================================================================
    LLM_API_BASE_URL: str = Field(default="https://api.openai.com/v1", description="LLM API base URL")
    LLM_MODEL_NAME: str = Field(default="o3-mini", description="LLM model name")

    # ================================================================
    # OPTIONAL: Embedding configuration (enables semantic caching when present)
    # ================================================================
    EMBEDDING_MODEL_API_KEY: str | None = Field(
        default=None, description="Embedding model API key (optional - enables semantic caching)"
    )
    EMBEDDING_MODEL_BASE_URL: str = Field(
        default="https://api.openai.com/v1", description="Embedding model API base URL"
    )
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-large", description="Embedding model name")

    # ================================================================
    # Database configuration with Docker Compose defaults
    # ================================================================
    DB_HOST: str = Field(default="postgres", description="Database host")
    DB_PORT: int = Field(default=5432, description="Database port")
    DB_NAME: str = Field(default="conversation_analysis", description="Database name")
    DB_USER: str = Field(default="cae_user", description="Database user")
    DB_SECRET: str = Field(default="cae_password", description="Database password")

    # ================================================================
    # Redis configuration with defaults
    # ================================================================
    REDIS_HOST: str = Field(default="redis", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_PASSWORD: str | None = Field(default=None, description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_POOL_SIZE: int = Field(default=20, description="Redis connection pool size")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, description="Redis max connections")

    # ================================================================
    # Feature toggles
    # ================================================================
    DISABLE_PROMETHEUS_METRICS: bool = Field(default=True, description="Disable Prometheus metrics collection")

    # ================================================================
    # Application settings
    # ================================================================
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Cache TTL in seconds")
    CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.85, description="Cache similarity threshold")
    CACHE_MAX_ENTRIES: int = Field(default=10000, description="Maximum cache entries")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LLM_TIMEOUT_SECONDS: int = Field(default=600, description="LLM request timeout")

    # ================================================================
    # Feature-flag outputs (computed automatically)
    # ================================================================
    enable_semantic_cache: bool = Field(default=False, description="Semantic caching enabled")
    enable_prometheus_metrics: bool = Field(default=True, description="Prometheus metrics enabled")

    def model_post_init(self, __context: dict | None) -> None:
        """
        Derive feature flags based on presence/override rules
        without mutating env-state.
        """
        # Semantic cache - ON if embedding API key supplied
        self.enable_semantic_cache = bool(self.EMBEDDING_MODEL_API_KEY)

        # Prometheus metrics - ON by default, off only when user opts-out
        self.enable_prometheus_metrics = not self.DISABLE_PROMETHEUS_METRICS

    def get_feature_summary(self) -> dict:
        """Get summary of enabled features."""
        return {
            "llm_configured": bool(self.LLM_API_KEY),
            "semantic_caching": self.enable_semantic_cache,
            "prometheus_metrics": self.enable_prometheus_metrics,
            "model": self.LLM_MODEL_NAME,
            "api_base": self.LLM_API_BASE_URL,
        }


app_settings = Config()
