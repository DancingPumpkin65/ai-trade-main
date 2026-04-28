from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "morocco-trading-agents"
    env: str = Field(default="dev", alias="ENV")
    data_dir: Path = Path("./data")
    db_path: Path = Field(default=Path("./data/trading.db"), alias="DB_PATH")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    drahmi_api_key: str | None = Field(default=None, alias="DRAHMI_API_KEY")
    drahmi_base_url: str = Field(default="https://api.drahmi.app/api", alias="DRAHMI_BASE_URL")
    drahmi_daily_limit: int = Field(default=500, alias="DRAHMI_DAILY_LIMIT")

    marketaux_api_key: str | None = Field(default=None, alias="MARKETAUX_API_KEY")
    marketaux_base_url: str = "https://api.marketaux.com/v1"

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:7b-instruct", alias="OLLAMA_MODEL")
    agent_llm_enabled: bool = Field(default=False, alias="AGENT_LLM_ENABLED")
    agent_llm_timeout: float = Field(default=20.0, alias="AGENT_LLM_TIMEOUT")

    chroma_persist_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    langgraph_checkpoint_path: Path = Field(
        default=Path("./data/langgraph-checkpoints.sqlite"),
        alias="LANGGRAPH_CHECKPOINT_PATH",
    )

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="morocco-trading-agents", alias="LANGSMITH_PROJECT")

    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")

    alpaca_api_key_id: str | None = Field(default=None, alias="ALPACA_API_KEY_ID")
    alpaca_api_secret_key: str | None = Field(default=None, alias="ALPACA_API_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL")
    alpaca_enabled: bool = Field(default=True, alias="ALPACA_ENABLED")
    alpaca_require_order_approval: bool = Field(default=True, alias="ALPACA_REQUIRE_ORDER_APPROVAL")
    alpaca_submit_orders: bool = Field(default=False, alias="ALPACA_SUBMIT_ORDERS")

    max_universe_candidates: int = 5
    max_returned_opportunities: int = 3


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    if not settings.database_url:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.langgraph_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
