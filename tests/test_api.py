from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from trading_agents.api.deps import get_services
from trading_agents.api.main import app
from trading_agents.core.models import GenerateSignalRequest, NewsChunk, RiskOutput, StockInfo
from trading_agents.graph import build as graph_build
from trading_agents.graph.technical_node import run_technical_agent as actual_run_technical_agent


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


def test_alpaca_preview_is_unmappable_after_approval_without_symbol_mapping(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
    services = get_services()

    async def fake_get_stock(symbol: str):
        return _volatile_stock(symbol)

    services.graph_service.drahmi_client.get_stock = fake_get_stock

    response = client.post("/signals/generate", json={"prompt": "Analyze ATW"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["alpaca_order_status"] == "UNMAPPABLE"
    assert payload["alpaca_order"]["source_symbol"] == "ATW"
    assert payload["alpaca_order"]["alpaca_symbol"] is None
    assert payload["alpaca_order"]["status"] == "UNMAPPABLE"

    alpaca_detail = client.get(f"/signals/{request_id}/alpaca-order")
    assert alpaca_detail.status_code == 200
    alpaca_payload = alpaca_detail.json()
    assert alpaca_payload["alpaca_order_status"] == "UNMAPPABLE"
    assert alpaca_payload["alpaca_order"]["reason"]


def test_alpaca_preview_is_prepared_after_approval_with_symbol_mapping(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
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

    approved = client.post(f"/signals/{request_id}/approve")
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["alpaca_order_status"] == "PREPARED"
    assert payload["alpaca_order"]["source_symbol"] == "ATW"
    assert payload["alpaca_order"]["alpaca_symbol"] == "SPY"
    assert payload["alpaca_order"]["side"] in {"buy", "sell"}
    assert payload["alpaca_order"]["type"] == "market"
    assert payload["alpaca_order"]["time_in_force"] == "day"
    assert payload["alpaca_order"]["preview_only"] is True
    assert payload["alpaca_order"]["submission_eligible"] is False

    detail = client.get(f"/signals/{request_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["alpaca_order_status"] == "PREPARED"
    assert detail_payload["alpaca_order"]["alpaca_symbol"] == "SPY"

    alpaca_detail = client.get(f"/signals/{request_id}/alpaca-order")
    assert alpaca_detail.status_code == 200
    alpaca_payload = alpaca_detail.json()
    assert alpaca_payload["alpaca_order_status"] == "PREPARED"
    assert alpaca_payload["alpaca_order"]["client_order_id"] == request_id


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


def test_technical_bias_mismatch_retries_then_recovers(tmp_path: Path, monkeypatch):
    _configure_env(tmp_path, monkeypatch)
    client = TestClient(app)
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
