import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import httpx

from trading_agents.api.deps import get_services
from trading_agents.api.main import app
from trading_agents.core.data.drahmi import DrahmiAuthError, DrahmiNotFoundError, DrahmiSchemaError
from trading_agents.core.models import GenerateSignalRequest, NewsChunk, RiskOutput, StockInfo
from trading_agents.graph import build as graph_build
from trading_agents.graph.technical_node import run_technical_agent as actual_run_technical_agent


class _AlpacaSubmissionResponse:
    def __init__(self, *, url: str, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.request = httpx.Request("POST", url)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "Request failed",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, json=self._payload),
            )


class _AlpacaSubmissionClient:
    def __init__(self, *, base_url: str, headers: dict, timeout: float, get_response, post_response):
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout
        self.get_response = get_response
        self.post_response = post_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get(self, path: str):
        return self.get_response

    def post(self, path: str, json: dict):
        return self.post_response


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


def _authenticate_client(client: TestClient, username: str = "operator", password: str = "test-pass-123") -> str:
    register = client.post("/auth/register", json={"username": username, "password": password})
    assert register.status_code == 200
    login = client.post("/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200
    token = login.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return token


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


def _trend_stock(
    symbol: str,
    name: str,
    *,
    start_price: float = 100.0,
    step: float = 1.2,
    last_volume: float = 2_000_000.0,
    bars: int = 30,
) -> StockInfo:
    history = []
    for idx in range(bars):
        close = start_price + step * idx
        history.append(
            {
                "date": f"2026-02-{idx + 1:02d}",
                "open": round(close * 0.995, 2),
                "high": round(close * 1.01, 2),
                "low": round(close * 0.99, 2),
                "close": round(close, 2),
                "volume": last_volume,
            }
        )
    return StockInfo(
        symbol=symbol,
        name=name,
        sector="Banks",
        market_cap=50_000_000_000,
        last_price=history[-1]["close"],
        last_volume=last_volume,
        high_52w=max(bar["close"] for bar in history),
        low_52w=min(bar["close"] for bar in history),
        ohlcv=history,
    )


def test_protected_routes_require_authentication(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    response = client.get("/history")
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_protected_routes_reject_invalid_token(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    response = client.get("/history", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication token."


def test_valid_token_allows_protected_access(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_generate_single_symbol_flow(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    response = client.post("/signals/generate", json={"prompt": "Analyze ATW with conservative risk"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request_intent"]["symbols_requested"] == ["ATW"]
    assert payload["signal_status"] == "COMPLETED"


def test_generate_returns_502_on_drahmi_auth_error(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    async def fake_get_stock(symbol: str):
        raise DrahmiAuthError("Drahmi request failed with status 401: unauthorized")

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 502
    assert "status 401" in response.json()["detail"]


def test_generate_returns_404_on_drahmi_not_found(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    async def fake_get_stock(symbol: str):
        raise DrahmiNotFoundError("Drahmi request failed with status 404: not found")

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 404
    assert "status 404" in response.json()["detail"]


def test_generate_returns_502_on_drahmi_schema_error_and_persists_failure(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    async def fake_get_stock(symbol: str):
        raise DrahmiSchemaError("/stocks/ATW field 'last_price' must be numeric.")

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 502
    assert "last_price" in response.json()["detail"]

    history = client.get("/history")
    assert history.status_code == 200
    failed_records = [record for record in history.json() if record["status"] == "FAILED"]
    assert failed_records
    request_id = failed_records[0]["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    failure = detail.json()["failure"]
    assert failure["error_type"] == "drahmi_schema_error"
    assert failure["status_code"] == 502


def test_generate_universe_scan(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    response = client.post("/signals/generate", json={"prompt": "I have 100,000 MAD. What are the best possible trades this week?"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request_intent"]["request_mode"] == "UNIVERSE_SCAN"
    assert "opportunity_list" in payload
    assert "universe_scan_candidates" in payload


def test_live_generate_stream_starts_request_and_emits_pipeline_events(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)

    client = TestClient(app)
    _authenticate_client(client)
    seen_events: list[tuple[str, dict]] = []
    current_event: str | None = None

    with client.stream("GET", "/signals/generate/stream", params={"prompt": "Analyze ATW with conservative risk"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                payload = json.loads(line.split(":", 1)[1].strip())
                seen_events.append((current_event, payload))
                if current_event == "pipeline_complete":
                    break

    event_names = [event_name for event_name, _ in seen_events]
    assert "request_started" in event_names
    assert "pipeline_start" in event_names
    assert "pipeline_complete" in event_names

    request_started_payload = next(payload for event_name, payload in seen_events if event_name == "request_started")
    request_id = request_started_payload["request_id"]
    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    assert detail.json()["signal_status"] == "COMPLETED"


def test_order_approval_flow_marks_prepared_preview_as_approved(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["signal_status"] == "COMPLETED"
    assert detail_payload["order_approval_required"] is True
    assert detail_payload["alpaca_order_status"] == "PREPARED"
    assert detail_payload["analysis_warnings"]

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["status"] == "COMPLETED"
    assert payload["final_signal"]["request_id"] == request_id
    assert payload["alpaca_order_status"] == "APPROVED"


def test_alpaca_preview_is_unmappable_after_analysis_without_symbol_mapping(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["alpaca_order_status"] == "UNMAPPABLE"
    assert payload["alpaca_order"]["source_symbol"] == "ATW"
    assert payload["alpaca_order"]["alpaca_symbol"] is None
    assert payload["alpaca_order"]["status"] == "UNMAPPABLE"
    assert payload["order_approval_required"] is False

    alpaca_detail = client.get(f"/signals/{request_id}/alpaca-order")
    assert alpaca_detail.status_code == 200
    alpaca_payload = alpaca_detail.json()
    assert alpaca_payload["alpaca_order_status"] == "UNMAPPABLE"
    assert alpaca_payload["alpaca_order"]["reason"]


def test_alpaca_preview_is_prepared_after_analysis_with_symbol_mapping(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    def fake_run_risk_agent(*, symbol, capital, sentiment_output, technical_output, technical_features, request_intent):
        return RiskOutput(
            action="BUY",
            position_size_pct=0.03,
            position_value_mad=capital * 0.03,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            risk_score=0.72,
            volatility_estimate=0.65,
            rationale="Forced BUY risk output for approval-preview integration test.",
        )

    services.graph_service.drahmi_client.get_stock = fake_get_stock
    monkeypatch.setattr(graph_build, "run_risk_agent", fake_run_risk_agent)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["alpaca_order_status"] == "PREPARED"
    assert payload["alpaca_order"]["source_symbol"] == "ATW"
    assert payload["alpaca_order"]["alpaca_symbol"] == "SPY"
    assert payload["alpaca_order"]["side"] in {"buy", "sell"}
    assert payload["alpaca_order"]["type"] == "market"
    assert payload["alpaca_order"]["time_in_force"] == "day"
    assert payload["alpaca_order"]["preview_only"] is True
    assert payload["order_approval_required"] is True
    assert payload["alpaca_order"]["submission_eligible"] is False

    alpaca_detail = client.get(f"/signals/{request_id}/alpaca-order")
    assert alpaca_detail.status_code == 200
    alpaca_payload = alpaca_detail.json()
    assert alpaca_payload["alpaca_order_status"] == "PREPARED"
    assert alpaca_payload["alpaca_order"]["client_order_id"] == request_id


def test_order_reject_flow_marks_prepared_preview_as_rejected(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    rejected = client.post(f"/signals/{request_id}/reject")
    assert rejected.status_code == 200
    payload = rejected.json()
    assert payload["status"] == "COMPLETED"
    assert payload["final_signal"] is not None
    assert payload["alpaca_order_status"] == "REJECTED"


def test_order_approval_endpoint_rejects_unmappable_preview(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 400


def test_full_access_mode_auto_approves_prepared_alpaca_command(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv("ALPACA_REQUIRE_ORDER_APPROVAL", "false")
    get_services.cache_clear()
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    def fake_run_risk_agent(*, symbol, capital, sentiment_output, technical_output, technical_features, request_intent):
        return RiskOutput(
            action="BUY",
            position_size_pct=0.03,
            position_value_mad=capital * 0.03,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            risk_score=0.72,
            volatility_estimate=0.65,
            rationale="Forced BUY risk output for auto-approval integration test.",
        )

    services.graph_service.drahmi_client.get_stock = fake_get_stock
    monkeypatch.setattr(graph_build, "run_risk_agent", fake_run_risk_agent)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["alpaca_order_status"] == "APPROVED"
    assert payload["order_approval_required"] is False
    assert payload["alpaca_order"]["submission_eligible"] is False

    events = services.stream_events(request_id)
    assert any(event["event_type"] == "alpaca_order_auto_approved" for event in events)


def test_order_approval_submits_to_alpaca_when_enabled(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv("ALPACA_SUBMIT_ORDERS", "true")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "paper-secret")
    get_services.cache_clear()
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    get_response = _AlpacaSubmissionResponse(
        url="https://paper-api.alpaca.markets/v2/assets/SPY",
        status_code=200,
        payload={"symbol": "SPY", "status": "active", "tradable": True, "fractionable": True},
    )
    post_response = _AlpacaSubmissionResponse(
        url="https://paper-api.alpaca.markets/v2/orders",
        status_code=200,
        payload={"id": "alpaca-submit-1", "status": "accepted"},
    )
    services.alpaca_preview_service.client_factory = lambda **kwargs: _AlpacaSubmissionClient(
        get_response=get_response,
        post_response=post_response,
        **kwargs,
    )

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    def fake_run_risk_agent(*, symbol, capital, sentiment_output, technical_output, technical_features, request_intent):
        return RiskOutput(
            action="BUY",
            position_size_pct=0.03,
            position_value_mad=capital * 0.03,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            risk_score=0.72,
            volatility_estimate=0.65,
            rationale="Forced BUY risk output for broker submit integration test.",
        )

    services.graph_service.drahmi_client.get_stock = fake_get_stock
    monkeypatch.setattr(graph_build, "run_risk_agent", fake_run_risk_agent)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["alpaca_order_status"] == "APPROVED"
    assert payload["alpaca_order"]["preview_only"] is False
    assert payload["alpaca_order"]["submission_eligible"] is True
    assert payload["alpaca_order"]["broker_order_id"] == "alpaca-submit-1"
    assert payload["alpaca_order"]["broker_order_status"] == "accepted"
    assert payload["alpaca_order"]["broker_submission_mode"] == "paper"


def test_full_access_mode_auto_submits_when_enabled(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv("ALPACA_REQUIRE_ORDER_APPROVAL", "false")
    monkeypatch.setenv("ALPACA_SUBMIT_ORDERS", "true")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "paper-secret")
    get_services.cache_clear()
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    get_response = _AlpacaSubmissionResponse(
        url="https://paper-api.alpaca.markets/v2/assets/SPY",
        status_code=200,
        payload={"symbol": "SPY", "status": "active", "tradable": True, "fractionable": True},
    )
    post_response = _AlpacaSubmissionResponse(
        url="https://paper-api.alpaca.markets/v2/orders",
        status_code=200,
        payload={"id": "alpaca-submit-2", "status": "accepted"},
    )
    services.alpaca_preview_service.client_factory = lambda **kwargs: _AlpacaSubmissionClient(
        get_response=get_response,
        post_response=post_response,
        **kwargs,
    )

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    def fake_run_risk_agent(*, symbol, capital, sentiment_output, technical_output, technical_features, request_intent):
        return RiskOutput(
            action="BUY",
            position_size_pct=0.03,
            position_value_mad=capital * 0.03,
            stop_loss_pct=0.05,
            take_profit_pct=0.08,
            risk_score=0.72,
            volatility_estimate=0.65,
            rationale="Forced BUY risk output for auto-submit integration test.",
        )

    services.graph_service.drahmi_client.get_stock = fake_get_stock
    monkeypatch.setattr(graph_build, "run_risk_agent", fake_run_risk_agent)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["alpaca_order_status"] == "APPROVED"
    assert payload["alpaca_order"]["preview_only"] is False
    assert payload["alpaca_order"]["submission_eligible"] is True
    assert payload["alpaca_order"]["broker_order_id"] == "alpaca-submit-2"


def test_technical_bias_mismatch_retries_then_recovers(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    calls = {"count": 0}

    def mismatch_once(stock_info, mismatch_feedback=None):
        calls["count"] += 1
        output, features = actual_run_technical_agent(stock_info, mismatch_feedback=mismatch_feedback)
        if calls["count"] == 1:
            flipped = "BEARISH" if features["directional_bias"] == "BULLISH" else "BULLISH"
            output = output.model_copy(update={"directional_bias": flipped, "trend_summary": "Bias intentionally mismatched for test."})
        return output, features

    monkeypatch.setattr(graph_build, "run_technical_agent", mismatch_once)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    saved_state = services.storage.get_saved_state(request_id)
    assert saved_state is not None
    assert saved_state["technical_retry_count"] == 1
    events = services.stream_events(request_id)
    assert any(event["event_type"] == "technical_retry" for event in events)


def test_technical_bias_mismatch_caps_and_force_corrects(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    def always_mismatch(stock_info, mismatch_feedback=None):
        output, features = actual_run_technical_agent(stock_info, mismatch_feedback=mismatch_feedback)
        flipped = "BEARISH" if features["directional_bias"] == "BULLISH" else "BULLISH"
        output = output.model_copy(update={"directional_bias": flipped, "trend_summary": "Bias intentionally mismatched for cap test."})
        return output, features

    monkeypatch.setattr(graph_build, "run_technical_agent", always_mismatch)

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    saved_state = services.storage.get_saved_state(request_id)
    assert saved_state is not None
    assert saved_state["technical_retry_count"] == 2
    assert saved_state["technical_output"]["directional_bias"] == saved_state["technical_features"]["directional_bias"]
    assert saved_state["errors"]


def test_agent_scratchpads_and_tool_events_are_recorded(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW with conservative risk"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    saved_state = services.storage.get_saved_state(request_id)
    assert saved_state is not None
    assert saved_state["sentiment_messages"]
    assert saved_state["technical_messages"]
    assert saved_state["risk_messages"]
    assert saved_state["coordinator_messages"]

    events = services.stream_events(request_id)
    event_types = [event["event_type"] for event in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    tool_names = {
        event["payload"]["tool"]
        for event in events
        if event["event_type"] == "tool_call" and "tool" in event["payload"]
    }
    assert "get_request_intent" in tool_names
    assert "get_technical_features" in tool_names
    assert "get_symbol" in tool_names
    assert "get_policy_note" in tool_names


def test_agent_iteration_cap_falls_back_safely(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.max_agent_iterations = 1

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    saved_state = services.storage.get_saved_state(request_id)
    assert saved_state is not None
    assert any("iteration cap" in error.lower() for error in saved_state["errors"])
    assert payload["final_signal"]["action"] == "HOLD"


def test_universe_scan_ranking_prefers_fresh_notice_candidate(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    services = get_services()
    intent = services.intent_parser.parse(
        GenerateSignalRequest(prompt="I have 100,000 MAD. What are the best possible trades this week?")
    )
    policy = services.graph_service.policy_engine.build(intent)
    now = datetime.now(timezone.utc)
    atw = _trend_stock("ATW", "Attijariwafa Bank", last_volume=2_500_000.0, step=1.8)
    iam = _trend_stock("IAM", "Maroc Telecom", last_volume=300_000.0, step=0.4)

    def fake_search_news(query, top_k=8, filters=None, metadata=None):
        if "ATW" in query:
            return [
                NewsChunk(
                    chunk_id="issuer-atw",
                    text="Attijariwafa Bank annonce un dividende 2025 et une convocation AGO.",
                    source="Casablanca Bourse Issuer Publication",
                    published_at=now - timedelta(days=1),
                    similarity_score=0.95,
                    url="https://example.com/atw.pdf",
                    metadata={"doc_type": "issuer_publication", "ticker": "ATW"},
                ),
                NewsChunk(
                    chunk_id="notice-atw",
                    text="ATW avis de mise en paiement du dividende.",
                    source="Casablanca Bourse PDF",
                    published_at=now - timedelta(days=2),
                    similarity_score=0.9,
                    url="https://example.com/atw-notice.pdf",
                    metadata={"doc_type": "corporate_notices", "ticker": "ATW"},
                ),
            ]
        return [
            NewsChunk(
                chunk_id="news-iam",
                text="Maroc Telecom information generale plus ancienne.",
                source="Sample Feed",
                published_at=now - timedelta(days=18),
                similarity_score=0.6,
                url="https://example.com/iam",
                metadata={"doc_type": "news", "ticker": "IAM"},
            )
        ]

    services.graph_service.retriever.search_news = fake_search_news
    atw_candidate, atw_rejection = services.graph_service._rank_universe_candidate(intent, atw, policy)
    iam_candidate, iam_rejection = services.graph_service._rank_universe_candidate(intent, iam, policy)

    assert atw_rejection is None
    assert atw_candidate is not None
    assert iam_candidate is None or atw_candidate.score > iam_candidate.score
    assert any("avis/publication" in reason.lower() for reason in atw_candidate.reasons)


def test_universe_scan_threshold_rejects_weak_candidates_before_deep_eval(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    services = get_services()
    intent = services.intent_parser.parse(
        GenerateSignalRequest(prompt="What are the best possible trades this week?", capital=100000)
    )

    async def fake_ingest_news(symbol=None):
        return None

    async def fake_list_stocks():
        return [
            _trend_stock("WEAK1", "Weak One", last_volume=1000.0, step=-0.2),
            _trend_stock("WEAK2", "Weak Two", last_volume=500.0, step=-0.1),
        ]

    def fake_search_news(query, top_k=8, filters=None, metadata=None):
        return []

    def should_not_run_symbol(*args, **kwargs):
        raise AssertionError("Detailed symbol evaluation should not run for rejected candidates.")

    services.graph_service.ingest_news = fake_ingest_news
    services.graph_service.drahmi_client.list_stocks = fake_list_stocks
    services.graph_service.retriever.search_news = fake_search_news
    services.graph_service._run_symbol_legacy = should_not_run_symbol

    result = services.graph_service._run_universe_scan(intent)
    assert result.top_opportunities == []
    assert result.rejected_candidates_summary
    assert any("insuffisant" in item.lower() or "liquidite" in item.lower() for item in result.rejected_candidates_summary)


def test_universe_scan_candidates_are_persisted_with_ranking_statuses(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    strong = _trend_stock("ATW", "Attijariwafa Bank", last_volume=2_400_000.0, step=1.7)
    weak = _trend_stock("WEAK", "Weak Name", last_volume=500.0, step=-0.2)

    async def fake_ingest_news(symbol=None):
        return None

    async def fake_list_stocks():
        return [strong, weak]

    async def fake_get_stock(symbol: str):
        return {"ATW": strong, "WEAK": weak}[symbol.upper()]

    def fake_search_news(query, top_k=8, filters=None, metadata=None):
        if "ATW" in query:
            return [
                NewsChunk(
                    chunk_id="atw-1",
                    text="ATW annonce un dividende et une hausse de resultat.",
                    source="Casablanca Bourse PDF",
                    published_at=datetime.now(timezone.utc) - timedelta(days=1),
                    similarity_score=0.95,
                    url="https://example.com/atw-1",
                    metadata={"doc_type": "corporate_notices", "ticker": "ATW"},
                )
            ]
        return []

    services.graph_service.ingest_news = fake_ingest_news
    services.graph_service.drahmi_client.list_stocks = fake_list_stocks
    services.graph_service.drahmi_client.get_stock = fake_get_stock
    services.graph_service.retriever.search_news = fake_search_news

    response = client.post("/signals/generate", json={"prompt": "I have 100,000 MAD. What are the best possible trades this week?"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    candidates = payload["universe_scan_candidates"]
    assert candidates
    assert any(candidate["selected_for_deep_eval"] for candidate in candidates)
    assert any(candidate["evaluation_status"] == "ACCEPTED_OPPORTUNITY" for candidate in candidates)
    assert any(candidate["evaluation_status"] in {"REJECTED_PRE_FILTER", "REJECTED_AFTER_DEEP_EVAL"} for candidate in candidates)
    ranked_positions = [
        candidate["rank_position"]
        for candidate in candidates
        if candidate["rank_position"] is not None
    ]
    assert ranked_positions == sorted(ranked_positions)


def test_universe_opportunity_order_flow_is_scoped_per_symbol(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()
    services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

    strong = _trend_stock("ATW", "Attijariwafa Bank", last_volume=2_400_000.0, step=1.7)
    weak = _trend_stock("IAM", "Maroc Telecom", last_volume=500_000.0, step=0.1)

    async def fake_ingest_news(symbol=None):
        return None

    async def fake_list_stocks():
        return [strong, weak]

    async def fake_get_stock(symbol: str):
        return {"ATW": strong, "IAM": weak}[symbol.upper()]

    def fake_search_news(query, top_k=8, filters=None, metadata=None):
        if "ATW" in query:
            return [
                NewsChunk(
                    chunk_id="atw-1",
                    text="ATW annonce un dividende et une hausse de resultat.",
                    source="Casablanca Bourse PDF",
                    published_at=datetime.now(timezone.utc) - timedelta(days=1),
                    similarity_score=0.95,
                    url="https://example.com/atw-1",
                    metadata={"doc_type": "corporate_notices", "ticker": "ATW"},
                )
            ]
        return []

    services.graph_service.ingest_news = fake_ingest_news
    services.graph_service.drahmi_client.list_stocks = fake_list_stocks
    services.graph_service.drahmi_client.get_stock = fake_get_stock
    services.graph_service.retriever.search_news = fake_search_news

    response = client.post("/signals/generate", json={"prompt": "I have 100,000 MAD. What are the best possible trades this week?"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["opportunity_list"]["top_opportunities"]
    assert payload["opportunity_alpaca_orders"] == {}

    opportunity_order = client.get(f"/signals/{request_id}/opportunities/ATW/alpaca-order")
    assert opportunity_order.status_code == 200
    order_payload = opportunity_order.json()
    assert order_payload["symbol"] == "ATW"
    assert order_payload["alpaca_order_status"] == "PREPARED"
    assert order_payload["order_approval_required"] is True

    approved = client.post(f"/signals/{request_id}/opportunities/ATW/approve")
    assert approved.status_code == 200
    approved_payload = approved.json()
    assert approved_payload["alpaca_order_status"] == "APPROVED"
    assert approved_payload["order_approval_required"] is False

    refreshed_detail = client.get(f"/signals/{request_id}")
    assert refreshed_detail.status_code == 200
    refreshed_payload = refreshed_detail.json()
    assert refreshed_payload["opportunity_alpaca_orders"]["ATW"]["status"] == "APPROVED"


def test_universe_opportunity_order_is_unmappable_without_mapping(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    _authenticate_client(client)
    services = get_services()

    strong = _trend_stock("ATW", "Attijariwafa Bank", last_volume=2_400_000.0, step=1.7)

    async def fake_ingest_news(symbol=None):
        return None

    async def fake_list_stocks():
        return [strong]

    async def fake_get_stock(symbol: str):
        return strong

    def fake_search_news(query, top_k=8, filters=None, metadata=None):
        return [
            NewsChunk(
                chunk_id="atw-1",
                text="ATW annonce un dividende et une hausse de resultat.",
                source="Casablanca Bourse PDF",
                published_at=datetime.now(timezone.utc) - timedelta(days=1),
                similarity_score=0.95,
                url="https://example.com/atw-1",
                metadata={"doc_type": "corporate_notices", "ticker": "ATW"},
            )
        ]

    services.graph_service.ingest_news = fake_ingest_news
    services.graph_service.drahmi_client.list_stocks = fake_list_stocks
    services.graph_service.drahmi_client.get_stock = fake_get_stock
    services.graph_service.retriever.search_news = fake_search_news

    response = client.post("/signals/generate", json={"prompt": "Best trades this week"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    opportunity_order = client.get(f"/signals/{request_id}/opportunities/ATW/alpaca-order")
    assert opportunity_order.status_code == 200
    payload = opportunity_order.json()
    assert payload["alpaca_order_status"] == "UNMAPPABLE"
    assert payload["order_approval_required"] is False

    approved = client.post(f"/signals/{request_id}/opportunities/ATW/approve")
    assert approved.status_code == 400
