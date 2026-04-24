from pathlib import Path

from fastapi.testclient import TestClient

from trading_agents.api.deps import get_services
from trading_agents.api.main import app
from trading_agents.core.models import StockInfo


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


def _volatile_stock(symbol: str = "ATW") -> StockInfo:
    closes = [100, 135, 82, 140, 78, 145, 74, 150, 70, 155, 68, 160, 66, 162, 64, 165, 62, 168, 60, 170, 58, 172]
    history = []
    for idx, close in enumerate(closes, start=1):
        history.append(
            {
                "date": f"2026-01-{idx:02d}",
                "open": close * 0.96,
                "high": close * 1.04,
                "low": close * 0.94,
                "close": close,
                "volume": 10000 + idx * 250,
            }
        )
    return StockInfo(
        symbol=symbol,
        name="Attijariwafa Bank",
        sector="Banks",
        market_cap=98_000_000_000,
        last_price=closes[-1],
        last_volume=history[-1]["volume"],
        high_52w=max(closes),
        low_52w=min(closes),
        ohlcv=history,
    )


def test_generate_single_symbol_flow(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

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
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    response = client.post("/signals/generate", json={"prompt": "I have 100,000 MAD. What are the best possible trades this week?"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request_intent"]["request_mode"] == "UNIVERSE_SCAN"
    assert "opportunity_list" in payload


def test_human_review_approve_flow(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    services = get_services()

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    assert detail.json()["signal_status"] == "WAITING_HUMAN"

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["status"] == "COMPLETED"
    assert payload["final_signal"]["request_id"] == request_id
    assert payload["alpaca_order_status"] in {"UNMAPPABLE", "PREPARED"}


def test_human_review_reject_flow(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    services = get_services()

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    rejected = client.post(f"/signals/{request_id}/reject")
    assert rejected.status_code == 200
    payload = rejected.json()
    assert payload["status"] == "REJECTED"
    assert payload["final_signal"] is None
