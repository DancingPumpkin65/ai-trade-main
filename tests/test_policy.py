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
