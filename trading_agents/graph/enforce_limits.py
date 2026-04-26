from __future__ import annotations

from datetime import datetime, timezone

from trading_agents.core.models import CoordinatorOutput, MarketMode, TradingSignal
from trading_agents.graph.helpers import (
    is_fixing_market,
    market_mode_daily_limit,
    market_mode_dynamic_reservation_limit,
    market_mode_static_reservation_limit,
    normalize_market_mode,
)


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
    execution_warnings: list[str] = []
    max_projected_move = max(stop_loss_pct, take_profit_pct)
    dynamic_limit = market_mode_dynamic_reservation_limit(resolved_market_mode)
    static_limit = market_mode_static_reservation_limit(resolved_market_mode)
    if dynamic_limit is not None and max_projected_move > dynamic_limit:
        execution_warnings.append(
            "Avertissement: seuil de reservation dynamique susceptible d'etre atteint (pause de 5 minutes possible)."
        )
    if static_limit is not None and max_projected_move > static_limit:
        execution_warnings.append(
            "Avertissement: seuil de reservation statique susceptible d'etre atteint (reservation puis enchere possible)."
        )
    gap_warning = None
    if take_profit_pct > daily_limit:
        gap_warning = "Avertissement: exécution probable sur plusieurs séances — risque de gap important."
        execution_warnings.append(gap_warning)
    return TradingSignal(
        symbol=symbol,
        action=coordinator_output.action,
        position_size_pct=round(position_size_pct, 4),
        position_value_mad=position_value_mad,
        stop_loss_pct=round(stop_loss_pct, 4),
        take_profit_pct=round(take_profit_pct, 4),
        risk_score=coordinator_output.risk_score,
        gap_risk_warning=gap_warning,
        execution_warnings=execution_warnings,
        rationale_fr=coordinator_output.rationale_fr,
        confidence=coordinator_output.confidence,
        market_mode=resolved_market_mode,
        is_fixing_mode=resolved_is_fixing_mode,
        request_id=request_id,
        generated_at=datetime.now(timezone.utc),
    )
