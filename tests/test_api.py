from pathlib import Path

from fastapi.testclient import TestClient

from trading_agents.api.deps import get_services
from trading_agents.api.main import app


def test_generate_single_symbol_flow(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRAHMI_API_KEY", "")
    monkeypatch.setenv("MARKETAUX_API_KEY", "")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DRAHMI_DAILY_LIMIT", "500")
    monkeypatch.setenv("ALPACA_ENABLED", "true")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("DRAHMI_BASE_URL", "https://api.drahmi.app/api")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("MARKETAUX_API_KEY", "")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "trading.db"))
    get_services.cache_clear()

    client = TestClient(app)
    response = client.post("/signals/generate", json={"prompt": "Analyze ATW with conservative risk"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request_intent"]["symbols_requested"] == ["ATW"]
    assert payload["signal_status"] in {"COMPLETED", "WAITING_HUMAN"}


def test_generate_universe_scan(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRAHMI_API_KEY", "")
    monkeypatch.setenv("MARKETAUX_API_KEY", "")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "trading.db"))
    get_services.cache_clear()

    client = TestClient(app)
    response = client.post("/signals/generate", json={"prompt": "I have 100,000 MAD. What are the best possible trades this week?"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request_intent"]["request_mode"] == "UNIVERSE_SCAN"
    assert "opportunity_list" in payload
