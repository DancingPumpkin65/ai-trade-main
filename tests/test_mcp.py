from pathlib import Path

from trading_agents.api.deps import get_services


def _configure_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DRAHMI_API_KEY", "")
    monkeypatch.setenv("MARKETAUX_API_KEY", "")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "trading.db"))
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "langgraph-checkpoints.sqlite"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ALPACA_ENABLED", "true")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    get_services.cache_clear()


def test_mcp_registers_four_namespace_surface(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    services = get_services()

    assert services.graph_service.mcp.list_tools("sentiment") == [
        "get_market_data",
        "get_request_intent",
        "search_news",
    ]
    assert services.graph_service.mcp.list_tools("technical") == [
        "analyze_technical",
        "get_market_data",
        "get_request_intent",
    ]
    assert services.graph_service.mcp.list_tools("risk") == [
        "calculate_size",
        "get_request_intent",
        "get_sentiment_output",
        "get_technical_features",
        "get_technical_output",
    ]
    assert services.graph_service.mcp.list_tools("coordinator") == [
        "get_all_outputs",
        "get_policy_note",
        "get_request_intent",
        "get_symbol",
    ]
