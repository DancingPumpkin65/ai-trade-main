from __future__ import annotations

from trading_agents.core.models import TechnicalOutput
from trading_agents.graph.helpers import analyze_technical_features


def run_technical_agent(stock_info, mismatch_feedback: str | None = None) -> tuple[TechnicalOutput, dict]:
    features = analyze_technical_features(stock_info)
    trend_summary = (
        "Tendance haussière confirmée par EMA10 au-dessus de SMA20."
        if features.directional_bias == "BULLISH"
        else "Tendance baissière confirmée par EMA10 sous SMA20."
        if features.directional_bias == "BEARISH"
        else "Tendance neutre avec peu de différenciation entre EMA10 et SMA20."
    )
    if mismatch_feedback:
        trend_summary = f"{trend_summary} Vérification supplémentaire: {mismatch_feedback}"
    output = TechnicalOutput(
        directional_bias=features.directional_bias,
        trend_summary=trend_summary,
        momentum_summary=f"RSI14 à {features.rsi14:.2f}, lecture de momentum {'solide' if features.rsi14 > 55 else 'faible' if features.rsi14 < 45 else 'neutre'}.",
        volatility_summary=f"Volatilité annualisée estimée à {features.annualized_volatility:.2%}.",
        support_levels=features.support_levels,
        resistance_levels=features.resistance_levels,
        volatility_estimate=features.annualized_volatility,
        liquidity_comment=(
            "Liquidité correcte."
            if features.zero_volume_bar_count == 0 and stock_info.last_volume > 0
            else "Liquidité irrégulière, prudence requise."
        ),
        confidence=0.72,
    )
    return output, features.model_dump()
