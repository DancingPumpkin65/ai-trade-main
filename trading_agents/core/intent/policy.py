from __future__ import annotations

from dataclasses import dataclass

from trading_agents.core.models import RequestIntent, RequestMode, RiskPreference, TimeHorizon


@dataclass(slots=True)
class IntentPolicy:
    news_weight: float = 1.0
    technical_weight: float = 1.0
    notice_weight: float = 1.0
    volatility_penalty: float = 1.0
    volatility_ceiling: float = 0.75
    min_liquidity_mad: float = 100_000.0
    freshness_bonus: float = 0.0
    momentum_bonus: float = 0.0
    execution_patience: str = "standard"
    conservative_posture: bool = False
    horizon_label: str = "standard"
    coordinator_note: str = "Adopter une posture equilibree."


class IntentPolicyEngine:
    def build(self, intent: RequestIntent) -> IntentPolicy:
        policy = IntentPolicy()
        if intent.risk_preference == RiskPreference.CONSERVATIVE:
            policy.volatility_penalty = 1.35
            policy.technical_weight = 1.1
            policy.notice_weight = 1.05
            policy.volatility_ceiling = 0.45
            policy.min_liquidity_mad = 250_000.0
            policy.conservative_posture = True
            policy.execution_patience = "patient"
            policy.coordinator_note = "Priviligier des configurations prudentes et liquides."
        elif intent.risk_preference == RiskPreference.AGGRESSIVE:
            policy.news_weight = 1.15
            policy.notice_weight = 0.95
            policy.volatility_penalty = 0.85
            policy.volatility_ceiling = 0.9
            policy.min_liquidity_mad = 50_000.0
            policy.execution_patience = "fast"
            policy.coordinator_note = "Accepter davantage de volatilite si le signal reste exploitable."

        if intent.time_horizon == TimeHorizon.SHORT_TERM:
            policy.news_weight += 0.2
            policy.technical_weight += 0.15
            policy.freshness_bonus += 0.08
            policy.momentum_bonus += 0.05
            policy.horizon_label = "short-term"
            policy.coordinator_note += " Donner plus de poids aux catalyseurs recents."
        elif intent.time_horizon == TimeHorizon.INTRADAY:
            policy.news_weight += 0.1
            policy.technical_weight += 0.25
            policy.notice_weight += 0.05
            policy.freshness_bonus += 0.14
            policy.momentum_bonus += 0.1
            policy.min_liquidity_mad = max(policy.min_liquidity_mad, 500_000.0)
            policy.volatility_ceiling = min(policy.volatility_ceiling, 0.4)
            policy.execution_patience = "fast"
            policy.horizon_label = "intraday"
            policy.coordinator_note += " Exiger des catalyseurs tres recents et une liquidite superieure."
        elif intent.time_horizon == TimeHorizon.SWING:
            policy.technical_weight += 0.12
            policy.freshness_bonus += 0.03
            policy.momentum_bonus += 0.03
            policy.volatility_ceiling = max(policy.volatility_ceiling, 0.8)
            policy.min_liquidity_mad = min(policy.min_liquidity_mad, 80_000.0)
            policy.execution_patience = "patient"
            policy.horizon_label = "swing"
            policy.coordinator_note += " Tolerer un horizon plus large avec des catalyseurs moins immediats."

        if intent.request_mode == RequestMode.UNIVERSE_SCAN:
            policy.news_weight += 0.1
            policy.notice_weight += 0.05
        return policy
