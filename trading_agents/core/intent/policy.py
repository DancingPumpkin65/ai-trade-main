from __future__ import annotations

from dataclasses import dataclass

from trading_agents.core.models import RequestIntent, RequestMode, RiskPreference, TimeHorizon


@dataclass(slots=True)
class IntentPolicy:
    news_weight: float = 1.0
    technical_weight: float = 1.0
    volatility_penalty: float = 1.0
    conservative_posture: bool = False
    horizon_label: str = "standard"


class IntentPolicyEngine:
    def build(self, intent: RequestIntent) -> IntentPolicy:
        policy = IntentPolicy()
        if intent.risk_preference == RiskPreference.CONSERVATIVE:
            policy.volatility_penalty = 1.35
            policy.technical_weight = 1.1
            policy.conservative_posture = True
        elif intent.risk_preference == RiskPreference.AGGRESSIVE:
            policy.news_weight = 1.15
            policy.volatility_penalty = 0.85

        if intent.time_horizon == TimeHorizon.SHORT_TERM:
            policy.news_weight += 0.2
            policy.technical_weight += 0.15
            policy.horizon_label = "short-term"
        elif intent.time_horizon == TimeHorizon.INTRADAY:
            policy.news_weight += 0.1
            policy.technical_weight += 0.25
            policy.horizon_label = "intraday"
        elif intent.time_horizon == TimeHorizon.SWING:
            policy.technical_weight += 0.05
            policy.horizon_label = "swing"

        if intent.request_mode == RequestMode.UNIVERSE_SCAN:
            policy.news_weight += 0.1
        return policy
