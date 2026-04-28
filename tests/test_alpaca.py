from __future__ import annotations

import httpx

from trading_agents.core.broker.alpaca import AlpacaPreviewService
from trading_agents.core.models import AlpacaOrderIntent, AlpacaOrderStatus, TradingSignal


class FakeResponse:
    def __init__(self, *, url: str, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.request = httpx.Request("GET", url)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "Request failed",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, json=self._payload),
            )


class FakeClient:
    def __init__(
        self,
        *,
        base_url: str,
        headers: dict,
        timeout: float,
        response: FakeResponse | None = None,
        post_response: FakeResponse | None = None,
        exception: Exception | None = None,
        post_exception: Exception | None = None,
    ):
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout
        self.response = response
        self.post_response = post_response
        self.exception = exception
        self.post_exception = post_exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get(self, path: str):
        if self.exception is not None:
            raise self.exception
        if self.response is None:
            raise AssertionError(f"No fake response configured for {path}")
        return self.response

    def post(self, path: str, json: dict):
        if self.post_exception is not None:
            raise self.post_exception
        if self.post_response is None:
            raise AssertionError(f"No fake POST response configured for {path} with payload {json}")
        return self.post_response


def _signal(symbol: str = "ATW", action: str = "BUY") -> TradingSignal:
    return TradingSignal.model_validate(
        {
            "symbol": symbol,
            "action": action,
            "position_size_pct": 0.03,
            "position_value_mad": 3000.0,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.08,
            "risk_score": 0.6,
            "gap_risk_warning": None,
            "execution_warnings": [],
            "rationale_fr": "Test signal",
            "confidence": 0.7,
            "market_mode": "CONTINUOUS",
            "is_fixing_mode": False,
            "request_id": "req-123",
            "generated_at": "2026-04-27T00:00:00+00:00",
        }
    )


def test_prepare_preview_skips_validation_without_credentials():
    service = AlpacaPreviewService(enabled=True)
    service.register_symbol_mapping("ATW", "SPY")

    preview = service.prepare_preview(_signal())

    assert preview.status == AlpacaOrderStatus.PREPARED
    assert preview.reason == "Asset validation skipped because Alpaca credentials are not configured."


def test_prepare_preview_marks_missing_asset_as_unmappable():
    response = FakeResponse(
        url="https://paper-api.alpaca.markets/v2/assets/SPY",
        status_code=404,
        payload={"message": "not found"},
    )
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(response=response, **kwargs),
    )
    service.register_symbol_mapping("ATW", "SPY")

    preview = service.prepare_preview(_signal())

    assert preview.status == AlpacaOrderStatus.UNMAPPABLE
    assert "not found" in (preview.reason or "").lower()


def test_prepare_preview_marks_inactive_asset_as_unmappable():
    response = FakeResponse(
        url="https://paper-api.alpaca.markets/v2/assets/SPY",
        status_code=200,
        payload={"symbol": "SPY", "status": "inactive", "tradable": True, "fractionable": True},
    )
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(response=response, **kwargs),
    )
    service.register_symbol_mapping("ATW", "SPY")

    preview = service.prepare_preview(_signal())

    assert preview.status == AlpacaOrderStatus.UNMAPPABLE
    assert "not active" in (preview.reason or "").lower()


def test_prepare_preview_marks_non_fractionable_asset_as_unmappable():
    response = FakeResponse(
        url="https://paper-api.alpaca.markets/v2/assets/SPY",
        status_code=200,
        payload={"symbol": "SPY", "status": "active", "tradable": True, "fractionable": False},
    )
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(response=response, **kwargs),
    )
    service.register_symbol_mapping("ATW", "SPY")

    preview = service.prepare_preview(_signal())

    assert preview.status == AlpacaOrderStatus.UNMAPPABLE
    assert "not fractionable" in (preview.reason or "").lower()


def test_prepare_preview_marks_api_failures_as_unmappable():
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(exception=httpx.ConnectError("network down"), **kwargs),
    )
    service.register_symbol_mapping("ATW", "SPY")

    preview = service.prepare_preview(_signal())

    assert preview.status == AlpacaOrderStatus.UNMAPPABLE
    assert "unable to validate" in (preview.reason or "").lower()


def test_submit_order_records_broker_submission_metadata():
    post_response = FakeResponse(
        url="https://paper-api.alpaca.markets/v2/orders",
        status_code=200,
        payload={"id": "alpaca-order-1", "status": "accepted"},
    )
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(post_response=post_response, **kwargs),
    )

    order = service.approve_preview(
        AlpacaOrderIntent.model_validate(
            {
                "request_id": "req-123",
                "client_order_id": "req-123",
                "source_symbol": "ATW",
                "alpaca_symbol": "SPY",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "notional": 3000.0,
                "preview_only": True,
                "submission_eligible": False,
                "status": "PREPARED",
                "created_at": "2026-04-27T00:00:00+00:00",
            }
        ),
        submission_enabled=True,
    )

    submitted = service.submit_order(order)

    assert submitted.preview_only is False
    assert submitted.submission_eligible is True
    assert submitted.broker_order_id == "alpaca-order-1"
    assert submitted.broker_order_status == "accepted"
    assert submitted.broker_submission_mode == "paper"
    assert submitted.reason == "Submitted to Alpaca paper trading."


def test_submit_order_raises_on_broker_rejection():
    post_response = FakeResponse(
        url="https://paper-api.alpaca.markets/v2/orders",
        status_code=403,
        payload={"message": "account not authorized"},
    )
    service = AlpacaPreviewService(
        enabled=True,
        api_key_id="key",
        api_secret_key="secret",
        client_factory=lambda **kwargs: FakeClient(post_response=post_response, **kwargs),
    )

    order = service.approve_preview(
        AlpacaOrderIntent.model_validate(
            {
                "request_id": "req-123",
                "client_order_id": "req-123",
                "source_symbol": "ATW",
                "alpaca_symbol": "SPY",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "notional": 3000.0,
                "preview_only": True,
                "submission_eligible": False,
                "status": "PREPARED",
                "created_at": "2026-04-27T00:00:00+00:00",
            }
        ),
        submission_enabled=True,
    )

    try:
        service.submit_order(order)
        assert False, "Expected submission failure"
    except ValueError as exc:
        assert "submission failed" in str(exc).lower()
