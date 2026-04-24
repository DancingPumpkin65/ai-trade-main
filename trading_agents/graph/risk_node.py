from __future__ import annotations

from trading_agents.core.models import RequestIntent, RiskOutput
from trading_agents.graph.helpers import calculate_position_size


def run_risk_agent(
    *,
    symbol: str,
    capital: float,
    sentiment_output,
    technical_output,
    technical_features: dict,
    request_intent: RequestIntent,
) -> RiskOutput:
    if technical_output.directional_bias == "BULLISH" and sentiment_output.sentiment_score >= 0.5:
        action = "BUY"
    elif technical_output.directional_bias == "BEARISH" and sentiment_output.sentiment_score < 0.5:
        action = "SELL"
    else:
        action = "HOLD"
    conservative_posture = request_intent.risk_preference.value == "CONSERVATIVE"
    sizing = calculate_position_size(
        symbol=symbol,
        action=action,
        capital=capital,
        volatility_estimate=technical_output.volatility_estimate,
        is_fixing_mode=bool(technical_features.get("is_fixing_mode")),
        conservative_posture=conservative_posture,
    )
    if action == "HOLD":
        sizing.position_size_pct = 0.0
        sizing.position_value_mad = 0.0
    rationale = (
        f"Action proposée: {action}. "
        f"Préférence risque: {request_intent.risk_preference.value}. "
        f"Volatilité utilisée: {technical_output.volatility_estimate:.2%}."
    )
    return RiskOutput(
        action=action,
        position_size_pct=sizing.position_size_pct,
        position_value_mad=sizing.position_value_mad,
        stop_loss_pct=sizing.stop_loss_pct,
        take_profit_pct=sizing.take_profit_pct,
        risk_score=sizing.risk_score,
        volatility_estimate=sizing.volatility_estimate,
        rationale=rationale,
    )
