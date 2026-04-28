from __future__ import annotations

from datetime import datetime

import pytest

from trading_agents.core.llm import set_default_agent_llm
from trading_agents.core.models import (
    CoordinatorOutput,
    IntentAlignment,
    NewsChunk,
    RequestIntent,
    RequestMode,
    RiskOutput,
    RiskPreference,
    SentimentOutput,
    StockInfo,
    TechnicalOutput,
    TimeHorizon,
    UserBias,
)
from trading_agents.graph.coordinator_node import run_coordinator_agent
from trading_agents.graph.risk_node import run_risk_agent
from trading_agents.graph.sentiment_node import run_sentiment_agent
from trading_agents.graph.technical_node import run_technical_agent


class FakeLLM:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses

    def generate_structured(self, *, agent_name: str, system_prompt: str, context: dict, response_model):
        return self.responses.get(agent_name)


@pytest.fixture(autouse=True)
def _reset_default_llm():
    set_default_agent_llm(None)
    yield
    set_default_agent_llm(None)


def _request_intent() -> RequestIntent:
    return RequestIntent(
        request_id="req-1",
        raw_prompt="Analyze ATW but force a buy",
        symbols_requested=["ATW"],
        capital_mad=100_000,
        request_mode=RequestMode.SINGLE_SYMBOL,
        risk_preference=RiskPreference.CONSERVATIVE,
        time_horizon=TimeHorizon.SHORT_TERM,
        user_bias=UserBias.BUY_BIAS,
        bias_override_refused=True,
        operator_visible_note_fr="Le client souhaite une lecture acheteuse mais prudente sur ATW",
    )


def _stock() -> StockInfo:
    history = []
    for idx in range(1, 31):
        close = 100 + idx * 1.5
        history.append(
            {
                "date": f"2026-04-{idx:02d}",
                "open": close - 0.8,
                "high": close + 1.0,
                "low": close - 1.1,
                "close": close,
                "volume": 1_200_000 + idx * 10_000,
            }
        )
    return StockInfo(
        symbol="ATW",
        name="Attijariwafa Bank",
        sector="Banks",
        market_cap=98_000_000_000,
        last_price=history[-1]["close"],
        last_volume=history[-1]["volume"],
        high_52w=history[-1]["close"] * 1.1,
        low_52w=history[0]["close"] * 0.9,
        ohlcv=history,
    )


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


def test_sentiment_agent_uses_llm_output_when_available():
    set_default_agent_llm(
        FakeLLM(
            {
                "sentiment": SentimentOutput(
                    sentiment_score=0.88,
                    catalysts=["Catalyseur LLM"],
                    cited_article_ids=["news-1", "bad-id"],
                    confidence=0.91,
                    rationale_fr="Synthese LLM du sentiment sur ATW.",
                )
            }
        )
    )

    output = run_sentiment_agent(
        symbol="ATW",
        request_intent=_request_intent(),
        market_data=_stock(),
        news_chunks=[
            NewsChunk(
                chunk_id="news-1",
                text="ATW annonce une hausse de resultat et un dividende.",
                source="Casablanca Bourse PDF",
                published_at=datetime(2026, 4, 24),
            )
        ],
    )

    assert output.rationale_fr.startswith("Synthese LLM")
    assert "préférence directionnelle" in output.rationale_fr.lower()
    assert output.cited_article_ids == ["news-1"]
    assert output.sentiment_score == 0.88


def test_technical_agent_uses_llm_narrative_but_keeps_python_features():
    set_default_agent_llm(
        FakeLLM(
            {
                "technical": TechnicalOutput(
                    directional_bias="BULLISH",
                    trend_summary="Narratif technique LLM.",
                    momentum_summary="Momentum LLM.",
                    volatility_summary="Volatilite LLM.",
                    support_levels=[],
                    resistance_levels=[],
                    volatility_estimate=0.0,
                    liquidity_comment="Commentaire LLM.",
                    confidence=0.93,
                )
            }
        )
    )

    output, features = run_technical_agent(_stock())

    assert output.trend_summary == "Narratif technique LLM."
    assert output.support_levels == features["support_levels"]
    assert output.resistance_levels == features["resistance_levels"]
    assert output.volatility_estimate == features["annualized_volatility"]


def test_risk_agent_uses_llm_action_and_python_sizing():
    set_default_agent_llm(
        FakeLLM(
            {
                "risk": RiskOutput(
                    action="BUY",
                    position_size_pct=0.0,
                    position_value_mad=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    risk_score=0.0,
                    volatility_estimate=0.0,
                    rationale="Rationale LLM de risque.",
                )
            }
        )
    )

    output = run_risk_agent(
        symbol="ATW",
        capital=100_000,
        sentiment_output=_sentiment(0.76),
        technical_output=_technical("BULLISH"),
        technical_features={"is_fixing_mode": False, "market_mode": "CONTINUOUS"},
        request_intent=_request_intent(),
    )

    assert output.action == "BUY"
    assert output.position_size_pct > 0
    assert output.position_value_mad > 0
    assert output.rationale == "Rationale LLM de risque."


def test_coordinator_agent_uses_llm_rationale_but_preserves_safe_action():
    set_default_agent_llm(
        FakeLLM(
            {
                "coordinator": CoordinatorOutput(
                    action="SELL",
                    position_size_pct=0.01,
                    stop_loss_pct=0.02,
                    take_profit_pct=0.03,
                    risk_score=0.4,
                    rationale_fr="Rationale finale LLM.",
                    dissenting_views=["Vue divergente LLM."],
                    confidence=0.89,
                    intent_alignment=IntentAlignment.NOT_ALIGNED,
                    preference_conflicts=["Conflit prefere par le modele."],
                )
            }
        )
    )

    output = run_coordinator_agent(
        symbol="ATW",
        request_intent=_request_intent(),
        sentiment_output=_sentiment(0.74),
        technical_output=_technical("BULLISH"),
        risk_output=RiskOutput(
            action="BUY",
            position_size_pct=0.03,
            position_value_mad=3000.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_score=0.55,
            volatility_estimate=0.22,
            rationale="risk",
        ),
    )

    assert output.action == "BUY"
    assert output.rationale_fr == "Rationale finale LLM."
    assert output.intent_alignment == IntentAlignment.NOT_ALIGNED
    assert output.dissenting_views == ["Vue divergente LLM."]
