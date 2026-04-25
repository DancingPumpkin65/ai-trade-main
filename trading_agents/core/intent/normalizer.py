from __future__ import annotations

import json
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from trading_agents.core.models import RequestMode, RiskPreference, TimeHorizon, UserBias


class NormalizedIntentHints(BaseModel):
    symbols_requested: list[str] = Field(default_factory=list)
    capital_mad: float | None = None
    request_mode: RequestMode | None = None
    risk_preference: RiskPreference | None = None
    time_horizon: TimeHorizon | None = None
    user_bias: UserBias | None = None
    bias_override_refused: bool | None = None
    parser_confidence: float = 0.6
    intent_notes_en: str = ""
    ambiguity_reason: str = ""


class IntentNormalizer(Protocol):
    def normalize(self, prompt: str) -> NormalizedIntentHints | None: ...


class OllamaIntentNormalizer:
    def __init__(self, *, base_url: str, model: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def normalize(self, prompt: str) -> NormalizedIntentHints | None:
        system_prompt = (
            "You normalize ambiguous trading-user prompts into a strict JSON object. "
            "Focus only on intent fields, not market advice. "
            "Allowed values: "
            "request_mode in [SINGLE_SYMBOL, UNIVERSE_SCAN], "
            "risk_preference in [CONSERVATIVE, BALANCED, AGGRESSIVE], "
            "time_horizon in [INTRADAY, SHORT_TERM, SWING, UNSPECIFIED], "
            "user_bias in [NONE, BUY_BIAS, SELL_BIAS]. "
            "If direction is forced, set bias_override_refused=true. "
            "Return JSON only with keys: "
            "symbols_requested, capital_mad, request_mode, risk_preference, time_horizon, "
            "user_bias, bias_override_refused, parser_confidence, intent_notes_en, ambiguity_reason."
        )
        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "prompt": f"{system_prompt}\n\nUser prompt:\n{prompt}",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            if not raw:
                return None
            return NormalizedIntentHints.model_validate_json(raw)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return None
