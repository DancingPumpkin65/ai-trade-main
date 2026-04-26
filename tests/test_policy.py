from trading_agents.core.intent.policy import IntentPolicyEngine
from trading_agents.core.models import RequestIntent, RequestMode, RiskPreference, TimeHorizon


def test_policy_changes_weights_for_conservative_short_term():
    policy = IntentPolicyEngine().build(
        RequestIntent(
            request_id="x",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.CONSERVATIVE,
            time_horizon=TimeHorizon.SHORT_TERM,
        )
    )
    assert policy.conservative_posture is True
    assert policy.news_weight > 1.0
    assert policy.volatility_penalty > 1.0
    assert policy.min_liquidity_mad >= 250000
    assert policy.execution_patience == "patient"


def test_policy_intraday_demands_more_liquidity_and_freshness():
    policy = IntentPolicyEngine().build(
        RequestIntent(
            request_id="x",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.INTRADAY,
        )
    )
    assert policy.horizon_label == "intraday"
    assert policy.min_liquidity_mad >= 500000
    assert policy.freshness_bonus >= 0.14
    assert policy.volatility_ceiling <= 0.4
    assert policy.execution_patience == "fast"


def test_policy_swing_tolerates_wider_horizon_than_intraday():
    intraday = IntentPolicyEngine().build(
        RequestIntent(
            request_id="x",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.INTRADAY,
        )
    )
    swing = IntentPolicyEngine().build(
        RequestIntent(
            request_id="y",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.SWING,
        )
    )
    assert swing.horizon_label == "swing"
    assert swing.technical_weight > intraday.technical_weight - 0.2
    assert swing.min_liquidity_mad < intraday.min_liquidity_mad
    assert swing.volatility_ceiling > intraday.volatility_ceiling
    assert swing.execution_patience == "patient"


def test_coordinator_prompt_context_changes_with_horizon():
    engine = IntentPolicyEngine()
    intraday_context = engine.build_coordinator_prompt_context(
        RequestIntent(
            request_id="x",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.INTRADAY,
        )
    )
    swing_context = engine.build_coordinator_prompt_context(
        RequestIntent(
            request_id="y",
            symbols_requested=["ATW"],
            capital_mad=100000,
            request_mode=RequestMode.SINGLE_SYMBOL,
            risk_preference=RiskPreference.BALANCED,
            time_horizon=TimeHorizon.SWING,
        )
    )
    assert intraday_context["interpretation_frame"] == "Lecture systeme intraday"
    assert "liquidite immediate" in intraday_context["interpretation_focus"]
    assert swing_context["interpretation_frame"] == "Lecture systeme swing"
    assert "plusieurs seances" in swing_context["interpretation_focus"]
