"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Central configuration for the 151 Trading System."""

    # Application
    app_env: str = "development"
    app_debug: bool = True
    app_secret_key: str = "change-me-to-a-random-secret"
    app_log_level: str = "INFO"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_system"
    postgres_user: str = "trading"
    postgres_password: str = "changeme"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Data Providers
    yahoo_finance_enabled: bool = True
    fred_api_key: Optional[str] = None
    fred_enabled: bool = True
    fmp_api_key: Optional[str] = None
    fmp_enabled: bool = True
    polygon_api_key: Optional[str] = None
    polygon_enabled: bool = True
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "151TradingSystem/0.1"
    reddit_enabled: bool = True
    trading_economics_client_key: Optional[str] = None
    trading_economics_secret_key: Optional[str] = None
    trading_economics_enabled: bool = True
    b3_enabled: bool = True
    anbima_enabled: bool = True
    fiscal_ai_api_key: Optional[str] = None
    fiscal_ai_enabled: bool = True

    # Broker (Alpaca)
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_paper_mode: bool = True

    # Frontend
    next_public_api_url: str = "http://localhost:8000"
    next_public_ws_url: str = "ws://localhost:8000"

    # Observability
    prometheus_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
