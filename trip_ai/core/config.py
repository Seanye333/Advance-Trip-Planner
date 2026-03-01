from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = "claude-sonnet-4-6"

    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    database_url: str = Field(default="", alias="DATABASE_URL")

    max_optimization_seconds: int = Field(default=30, alias="MAX_OPTIMIZATION_SECONDS")
    monte_carlo_iterations: int = Field(default=10_000, alias="MONTE_CARLO_ITERATIONS")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


settings = Settings()
