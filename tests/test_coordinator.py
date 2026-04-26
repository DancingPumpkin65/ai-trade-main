from trading_agents.core.models import (
    RequestIntent,
    RequestMode,
    RiskOutput,
    RiskPreference,
    SentimentOutput,
    TechnicalOutput,
    TimeHorizon,
    UserBias,
)
from trading_agents.graph.coordinator_node import run_coordinator_agent


def _sentiment(score: float) -> SentimentOutput:
    return SentimentOutput(
        sentiment_score=score,
        catalysts=[],
        cited_article_ids=[],
        confidence=0.7,
        rationale_fr="sentiment",
    )


def _technical(bias: str) -> TechnicalOutput:
    return TechnicalOutput(
        directional_bias=bias,
        trend_summary="trend",
        momentum_summary="momentum",
        volatility_summary="volatility",
        support_levels=[100.0],
        resistance_levels=[110.0],
        volatility_estimate=0.22,
        liquidity_comment="liquide",
        confidence=0.8,
    )


def _risk(action: str) -> RiskOutput:
    return RiskOutput(
        action=action,
        position_size_pct=0.03,
        position_value_mad=3000.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_score=0.55,
        volatility_estimate=0.22,
        rationale="risk",
    )


def test_coordinator_rationale_uses_intraday_template():
    output = run_coordinator_agent(
        symbol="ATW",
        request_intent=RequestIntent(
            request_id="req-1",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.INTRADAY,
            operator_visible_note_fr="Analyse ATW en horizon intraday",
        ),
        sentiment_output=_sentiment(0.72),
        technical_output=_technical("BULLISH"),
        risk_output=_risk("BUY"),
    )
    assert "Lecture systeme intraday" in output.rationale_fr
    assert "liquidite immediate" in output.rationale_fr
    assert "Recommendation finale tres court terme" in output.rationale_fr


def test_coordinator_rationale_explains_forced_buy_conflict():
    output = run_coordinator_agent(
        symbol="ATW",
        request_intent=RequestIntent(
            request_id="req-2",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.CONSERVATIVE,
            time_horizon=TimeHorizon.SHORT_TERM,
            user_bias=UserBias.BUY_BIAS,
            bias_override_refused=True,
            operator_visible_note_fr="Le client souhaite une idee acheteuse sur ATW avec prudence",
        ),
        sentiment_output=_sentiment(0.42),
        technical_output=_technical("BEARISH"),
        risk_output=_risk("SELL"),
    )
    assert "Ecart avec la preference client" in output.rationale_fr
    assert "préférence haussière du client n'est pas retenue" in output.rationale_fr
    assert output.intent_alignment.value == "NOT_ALIGNED"
