from __future__ import annotations

from trading_agents.core.intent.policy import IntentPolicyEngine
from trading_agents.core.llm import get_default_agent_llm
from trading_agents.core.models import RequestIntent, RiskOutput
from trading_agents.graph.helpers import calculate_position_size


def _deterministic_risk_agent(
    *,
    symbol: str,
    capital: float,
    sentiment_output,
    technical_output,
    technical_features: dict,
    request_intent: RequestIntent,
) -> RiskOutput:
    policy = IntentPolicyEngine().build(request_intent)
    bullish_threshold = 0.5
    bearish_threshold = 0.5
    if policy.execution_patience == "patient":
        bullish_threshold = 0.55
        bearish_threshold = 0.45
    elif policy.execution_patience == "fast":
        bullish_threshold = 0.48
        bearish_threshold = 0.52

    if technical_output.directional_bias == "BULLISH" and sentiment_output.sentiment_score >= bullish_threshold:
        action = "BUY"
    elif technical_output.directional_bias == "BEARISH" and sentiment_output.sentiment_score <= bearish_threshold:
        action = "SELL"
    else:
        action = "HOLD"
    conservative_posture = policy.conservative_posture
    sizing = calculate_position_size(
        symbol=symbol,
        action=action,
        capital=capital,
        volatility_estimate=technical_output.volatility_estimate,
        is_fixing_mode=bool(technical_features.get("is_fixing_mode")),
        market_mode=technical_features.get("market_mode"),
        conservative_posture=conservative_posture,
    )
    if action == "HOLD":
        sizing.position_size_pct = 0.0
        sizing.position_value_mad = 0.0
    rationale = (
        f"Action proposée: {action}. "
        f"Préférence risque: {request_intent.risk_preference.value}. "
        f"Horizon applique: {policy.horizon_label}. "
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


def run_risk_agent(
    *,
    symbol: str,
    capital: float,
    sentiment_output,
    technical_output,
    technical_features: dict,
    request_intent: RequestIntent,
) -> RiskOutput:
    deterministic = _deterministic_risk_agent(
        symbol=symbol,
        capital=capital,
        sentiment_output=sentiment_output,
        technical_output=technical_output,
        technical_features=technical_features,
        request_intent=request_intent,
    )
    llm = get_default_agent_llm()
    if llm is None:
        return deterministic

    llm_output = llm.generate_structured(
        agent_name="risk",
        system_prompt=(
            "Recommend a trading action from the supplied sentiment, technical, and intent context. "
            "Use only BUY, SELL, or HOLD for the action. "
            "Hard risk numbers will be recomputed by Python, so focus on action quality and rationale."
        ),
        context={
            "symbol": symbol,
            "capital_mad": capital,
            "request_intent": request_intent.model_dump(mode="json"),
            "sentiment_output": sentiment_output.model_dump(mode="json"),
            "technical_output": technical_output.model_dump(mode="json"),
            "technical_features": technical_features,
            "deterministic_baseline": deterministic.model_dump(mode="json"),
        },
        response_model=RiskOutput,
    )
    if llm_output is None:
        return deterministic

    action = str(llm_output.action or deterministic.action).upper()
    if action not in {"BUY", "SELL", "HOLD"}:
        action = deterministic.action
    policy = IntentPolicyEngine().build(request_intent)
    sizing = calculate_position_size(
        symbol=symbol,
        action=action,
        capital=capital,
        volatility_estimate=technical_output.volatility_estimate,
        is_fixing_mode=bool(technical_features.get("is_fixing_mode")),
        market_mode=technical_features.get("market_mode"),
        conservative_posture=policy.conservative_posture,
    )
    position_size_pct = sizing.position_size_pct
    position_value_mad = sizing.position_value_mad
    if action == "HOLD":
        position_size_pct = 0.0
        position_value_mad = 0.0
    return deterministic.model_copy(
        update={
            "action": action,
            "position_size_pct": position_size_pct,
            "position_value_mad": position_value_mad,
            "stop_loss_pct": sizing.stop_loss_pct,
            "take_profit_pct": sizing.take_profit_pct,
            "risk_score": sizing.risk_score,
            "volatility_estimate": sizing.volatility_estimate,
            "rationale": llm_output.rationale or deterministic.rationale,
        }
    )
