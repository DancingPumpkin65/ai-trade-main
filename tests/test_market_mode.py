from trading_agents.core.data.drahmi import DrahmiClient
from trading_agents.core.models import CoordinatorOutput, MarketMode, StockInfo
from trading_agents.graph.enforce_limits import enforce_limits
from trading_agents.graph.helpers import (
    analyze_technical_features,
    market_mode_daily_limit,
    market_mode_dynamic_reservation_limit,
    market_mode_static_reservation_limit,
)


def _stock_with_mode(mode: MarketMode, *, price: float = 120.0, volume: float = 1_000_000.0) -> StockInfo:
    history = []
    for idx in range(30):
        close = price + idx * 0.5
        history.append(
            {
                "date": f"2026-03-{idx + 1:02d}",
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": volume,
            }
        )
    return StockInfo(
        symbol="ATW",
        name="Attijariwafa Bank",
        sector="Banks",
        market_cap=1.0,
        last_price=history[-1]["close"],
        last_volume=volume,
        high_52w=max(bar["close"] for bar in history),
        low_52w=min(bar["close"] for bar in history),
        market_mode=mode,
        market_metadata={"market_mode": mode.value.lower()},
        ohlcv=history,
    )


def test_drahmi_payload_extracts_fixing_market_mode():
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)
    stock = client._payload_to_stock(
        {
            "ticker": "ATW",
            "name": "Attijariwafa Bank",
            "sector": "Banks",
            "price": 520.0,
            "volume": 1_500_000,
            "market_mode": "fixing",
            "history": [],
        }
    )
    assert stock.market_mode == MarketMode.FIXING


def test_drahmi_payload_extracts_bond_market_mode():
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)
    stock = client._payload_to_stock(
        {
            "ticker": "BOND1",
            "name": "Moroccan Treasury Bond",
            "sector": "Fixed Income",
            "price": 102.0,
            "volume": 50_000,
            "instrument_type": "bond",
            "history": [],
        }
    )
    assert stock.market_mode == MarketMode.BOND


def test_technical_features_use_explicit_market_mode_not_price_heuristic():
    stock = _stock_with_mode(MarketMode.CONTINUOUS, price=60.0, volume=2_000_000.0)
    features = analyze_technical_features(stock)
    assert features.market_mode == MarketMode.CONTINUOUS
    assert features.is_fixing_mode is False


def test_enforce_limits_applies_fixing_haircut_from_market_mode():
    coordinator_output = CoordinatorOutput(
        action="BUY",
        position_size_pct=0.05,
        stop_loss_pct=0.09,
        take_profit_pct=0.12,
        risk_score=0.6,
        rationale_fr="Test",
        dissenting_views=[],
        confidence=0.7,
    )
    signal = enforce_limits(
        symbol="ATW",
        request_id="req-1",
        coordinator_output=coordinator_output,
        is_fixing_mode=False,
        market_mode=MarketMode.FIXING,
        capital=100_000.0,
    )
    assert signal.market_mode == MarketMode.FIXING
    assert signal.is_fixing_mode is True
    assert signal.position_size_pct == 0.04
    assert signal.stop_loss_pct == market_mode_daily_limit(MarketMode.FIXING)
    assert any("reservation dynamique" in item.lower() for item in signal.execution_warnings)
    assert any("reservation statique" in item.lower() for item in signal.execution_warnings)


def test_enforce_limits_preserves_bond_detection_without_fixing_haircut():
    coordinator_output = CoordinatorOutput(
        action="BUY",
        position_size_pct=0.05,
        stop_loss_pct=0.04,
        take_profit_pct=0.07,
        risk_score=0.4,
        rationale_fr="Bond test",
        dissenting_views=[],
        confidence=0.65,
    )
    signal = enforce_limits(
        symbol="BOND1",
        request_id="req-bond",
        coordinator_output=coordinator_output,
        is_fixing_mode=False,
        market_mode=MarketMode.BOND,
        capital=100_000.0,
    )
    assert signal.market_mode == MarketMode.BOND
    assert signal.is_fixing_mode is False
    assert signal.position_size_pct == 0.05
    assert signal.stop_loss_pct == 0.02
    assert signal.gap_risk_warning is not None
    assert len(signal.execution_warnings) == 1


def test_market_mode_daily_limit_uses_bond_limit():
    assert market_mode_daily_limit(MarketMode.BOND) == 0.02
    assert market_mode_daily_limit(MarketMode.FIXING) == 0.06
    assert market_mode_daily_limit(MarketMode.CONTINUOUS) == 0.10


def test_market_mode_reservation_limits_match_local_spec():
    assert market_mode_dynamic_reservation_limit(MarketMode.CONTINUOUS) == 0.03
    assert market_mode_dynamic_reservation_limit(MarketMode.FIXING) == 0.03
    assert market_mode_dynamic_reservation_limit(MarketMode.BOND) is None
    assert market_mode_static_reservation_limit(MarketMode.CONTINUOUS) == 0.06
    assert market_mode_static_reservation_limit(MarketMode.FIXING) == 0.04
    assert market_mode_static_reservation_limit(MarketMode.BOND) is None


def test_enforce_limits_adds_dynamic_but_not_static_warning_for_continuous_mid_range_move():
    coordinator_output = CoordinatorOutput(
        action="BUY",
        position_size_pct=0.04,
        stop_loss_pct=0.025,
        take_profit_pct=0.05,
        risk_score=0.35,
        rationale_fr="Continuous warning test",
        dissenting_views=[],
        confidence=0.6,
    )
    signal = enforce_limits(
        symbol="ATW",
        request_id="req-cont",
        coordinator_output=coordinator_output,
        is_fixing_mode=False,
        market_mode=MarketMode.CONTINUOUS,
        capital=100_000.0,
    )
    assert any("reservation dynamique" in item.lower() for item in signal.execution_warnings)
    assert not any("reservation statique" in item.lower() for item in signal.execution_warnings)
    assert signal.gap_risk_warning is None
