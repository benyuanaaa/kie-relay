"""Configuration management for kie.ai relay."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # kie.ai credentials
    kie_api_key: str = ""
    kie_api_base: str = "https://api.kie.ai"

    # Relay server settings
    host: str = "0.0.0.0"
    port: int = 5001

    # Polling settings (seconds)
    # Koyeb free tier has ~60s request timeout, so we poll faster
    poll_interval: float = 1.0
    poll_timeout: float = 55.0

    # Logging
    log_level: str = "INFO"

    # Master API key for the relay itself (optional)
    # If set, clients must provide this key; otherwise any key is accepted
    relay_api_key: Optional[str] = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
