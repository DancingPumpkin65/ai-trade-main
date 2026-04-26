from __future__ import annotations

from datetime import datetime, timezone

from trading_agents.core.models import CoordinatorOutput, MarketMode, TradingSignal
from trading_agents.graph.helpers import is_fixing_market, market_mode_daily_limit, normalize_market_mode


def enforce_limits(
    *,
    symbol: str,
    request_id: str,
    coordinator_output: CoordinatorOutput,
    is_fixing_mode: bool,
    capital: float,
    market_mode: MarketMode = MarketMode.UNKNOWN,
) -> TradingSignal:
    resolved_market_mode = MarketMode.FIXING if is_fixing_mode else normalize_market_mode(market_mode)
    daily_limit = market_mode_daily_limit(resolved_market_mode)
    resolved_is_fixing_mode = is_fixing_market(resolved_market_mode)
    stop_loss_pct = min(coordinator_output.stop_loss_pct, daily_limit)
    take_profit_pct = max(coordinator_output.take_profit_pct, stop_loss_pct * 1.5)
    position_size_pct = min(coordinator_output.position_size_pct, 0.05)
    if resolved_is_fixing_mode:
        position_size_pct *= 0.8
    position_value_mad = round(capital * position_size_pct, 2)
    gap_warning = None
    if take_profit_pct > daily_limit:
        gap_warning = "Avertissement: exécution probable sur plusieurs séances — risque de gap important."
    return TradingSignal(
        symbol=symbol,
        action=coordinator_output.action,
        position_size_pct=round(position_size_pct, 4),
        position_value_mad=position_value_mad,
        stop_loss_pct=round(stop_loss_pct, 4),
        take_profit_pct=round(take_profit_pct, 4),
        risk_score=coordinator_output.risk_score,
        gap_risk_warning=gap_warning,
        rationale_fr=coordinator_output.rationale_fr,
        confidence=coordinator_output.confidence,
        market_mode=resolved_market_mode,
        is_fixing_mode=resolved_is_fixing_mode,
        request_id=request_id,
        generated_at=datetime.now(timezone.utc),
    )
