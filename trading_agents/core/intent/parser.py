from __future__ import annotations

import re
import uuid

from trading_agents.core.intent.normalizer import IntentNormalizer, NormalizedIntentHints
from trading_agents.core.models import (
    GenerateSignalRequest,
    RequestIntent,
    RequestMode,
    RiskPreference,
    TimeHorizon,
    UserBias,
)
from trading_agents.core.observability import LangSmithIntentTracer


SYMBOL_PATTERN = re.compile(r"\b([A-Za-z]{2,5})\b")
CAPITAL_PATTERN = re.compile(r"(\d[\d\s,._]*)\s*(MAD|DH|dirhams?)", re.IGNORECASE)
ANALYZE_SYMBOL_PATTERN = re.compile(r"\b(?:analyze|analyse|study|review)\s+([A-Za-z]{2,5})\b", re.IGNORECASE)
STOPWORDS = {
    "WITH",
    "RISK",
    "THIS",
    "WEEK",
    "WHAT",
    "BEST",
    "POSSIBLE",
    "TRADES",
    "HAVE",
    "AND",
    "THE",
    "IDEA",
    "FORCE",
    "BUY",
    "SELL",
    "MAD",
}


class IntentParser:
    def __init__(
        self,
        *,
        normalizer: IntentNormalizer | None = None,
        tracer: LangSmithIntentTracer | None = None,
    ):
        self.normalizer = normalizer
        self.tracer = tracer

    def parse(self, payload: GenerateSignalRequest) -> RequestIntent:
        request_id = str(uuid.uuid4())
        raw_prompt = (payload.prompt or "").strip() or None
        prompt_text = f"{payload.symbol or ''} {payload.prompt or ''}".strip()
        deterministic_symbols = self._extract_symbols(payload.symbol, prompt_text)
        deterministic_capital = self._extract_capital(payload.capital, prompt_text)
        deterministic_risk = payload.risk_profile or self._extract_risk(prompt_text)
        deterministic_horizon = payload.time_horizon or self._extract_horizon(prompt_text)
        user_bias, refused = self._extract_bias(prompt_text)
        deterministic_mode = self._extract_mode(prompt_text, deterministic_symbols)
        parser_confidence = self._deterministic_confidence(prompt_text, deterministic_symbols, deterministic_mode)
        llm_hints = None
        if raw_prompt and self._should_use_llm_fallback(prompt_text, deterministic_symbols, deterministic_mode, parser_confidence):
            llm_hints = self.normalizer.normalize(raw_prompt) if self.normalizer is not None else None
            if self.tracer is not None:
                self.tracer.log_normalization(
                    prompt=raw_prompt,
                    deterministic={
                        "symbols_requested": deterministic_symbols,
                        "capital_mad": deterministic_capital,
                        "request_mode": deterministic_mode.value,
                        "risk_preference": deterministic_risk.value,
                        "time_horizon": deterministic_horizon.value,
                        "user_bias": user_bias.value,
                        "bias_override_refused": refused,
                        "parser_confidence": parser_confidence,
                    },
                    normalized=llm_hints.model_dump(mode="json") if llm_hints is not None else None,
                    metadata={"request_id": request_id},
                )

        symbols = deterministic_symbols or (llm_hints.symbols_requested if llm_hints else [])
        capital = deterministic_capital
        if payload.capital is None and llm_hints and llm_hints.capital_mad is not None:
            capital = llm_hints.capital_mad
        risk_preference = deterministic_risk if payload.risk_profile is not None or deterministic_risk != RiskPreference.BALANCED else (llm_hints.risk_preference if llm_hints and llm_hints.risk_preference is not None else deterministic_risk)
        time_horizon = deterministic_horizon if payload.time_horizon is not None or deterministic_horizon != TimeHorizon.UNSPECIFIED else (llm_hints.time_horizon if llm_hints and llm_hints.time_horizon is not None else deterministic_horizon)
        if user_bias == UserBias.NONE and llm_hints and llm_hints.user_bias is not None:
            user_bias = llm_hints.user_bias
        refused = refused or bool(llm_hints and llm_hints.bias_override_refused)
        request_mode = deterministic_mode if deterministic_symbols else (llm_hints.request_mode if llm_hints and llm_hints.request_mode is not None else deterministic_mode)
        if llm_hints is not None:
            parser_confidence = max(parser_confidence, min(max(llm_hints.parser_confidence, 0.0), 1.0))
        notes = []
        if request_mode == RequestMode.UNIVERSE_SCAN:
            notes.append("User asked for opportunity discovery across the Moroccan universe.")
        if risk_preference == RiskPreference.CONSERVATIVE:
            notes.append("Use conservative posture in ranking and coordinator narrative.")
        if time_horizon != TimeHorizon.UNSPECIFIED:
            notes.append(f"Horizon preference detected: {time_horizon.value}.")
        if refused:
            notes.append("User attempted to force direction; system must not obey.")
        if llm_hints and llm_hints.intent_notes_en:
            notes.append(llm_hints.intent_notes_en)
        if not symbols and request_mode != RequestMode.UNIVERSE_SCAN:
            raise ValueError("Request must include a symbol or a scannable universe request.")
        return RequestIntent(
            request_id=request_id,
            raw_prompt=raw_prompt,
            symbols_requested=symbols,
            capital_mad=capital,
            request_mode=request_mode,
            risk_preference=risk_preference,
            time_horizon=time_horizon,
            user_bias=user_bias,
            bias_override_refused=refused,
            intent_notes_en=" ".join(notes) or "Standard analysis request.",
            operator_visible_note_fr=self._operator_note_fr(risk_preference, time_horizon, user_bias, refused, request_mode),
            parser_confidence=parser_confidence,
            extraction_method="deterministic+llm" if llm_hints is not None else "deterministic",
        )

    def _should_use_llm_fallback(
        self,
        text: str,
        symbols: list[str],
        request_mode: RequestMode,
        parser_confidence: float,
    ) -> bool:
        if parser_confidence < 0.9:
            return True
        lowered = text.lower()
        if not symbols and request_mode == RequestMode.SINGLE_SYMBOL:
            return True
        if len(symbols) > 1:
            return True
        if any(token in lowered for token in ("something safe", "give me an idea", "find me a trade", "try a trade")):
            return True
        return False

    def _deterministic_confidence(self, text: str, symbols: list[str], request_mode: RequestMode) -> float:
        lowered = text.lower()
        if symbols and ANALYZE_SYMBOL_PATTERN.search(text):
            return 0.98
        if request_mode == RequestMode.UNIVERSE_SCAN and any(
            phrase in lowered
            for phrase in ("best possible trades", "best trades", "scan the market", "opportunities this week")
        ):
            return 0.95
        if len(symbols) > 1:
            return 0.55
        if symbols:
            return 0.82
        if request_mode == RequestMode.UNIVERSE_SCAN:
            return 0.72
        return 0.35

    def _extract_symbols(self, explicit_symbol: str | None, text: str) -> list[str]:
        if explicit_symbol:
            return [explicit_symbol.upper()]
        candidates: list[str] = []
        directive_match = ANALYZE_SYMBOL_PATTERN.search(text)
        if directive_match:
            candidates.append(directive_match.group(1).upper())
        for match in SYMBOL_PATTERN.finditer(text):
            token = match.group(1)
            if token.isupper():
                candidates.append(token.upper())
            elif len(token) <= 4 and token.lower() == token and directive_match and token.upper() == directive_match.group(1).upper():
                candidates.append(token.upper())
        filtered = [symbol for symbol in candidates if symbol not in STOPWORDS]
        unique: list[str] = []
        for symbol in filtered:
            if symbol not in unique:
                unique.append(symbol)
        return unique[:3]

    def _extract_capital(self, explicit_capital: float | None, text: str) -> float:
        if explicit_capital is not None:
            return float(explicit_capital)
        match = CAPITAL_PATTERN.search(text)
        if not match:
            return 100_000.0
        raw = match.group(1).replace(" ", "").replace(",", "").replace("_", "")
        return float(raw)

    def _extract_risk(self, text: str) -> RiskPreference:
        lowered = text.lower()
        if any(token in lowered for token in ("conservative", "prudent", "low risk")):
            return RiskPreference.CONSERVATIVE
        if any(token in lowered for token in ("aggressive", "speculative", "high risk")):
            return RiskPreference.AGGRESSIVE
        return RiskPreference.BALANCED

    def _extract_horizon(self, text: str) -> TimeHorizon:
        lowered = text.lower()
        if any(token in lowered for token in ("intraday", "today", "session")):
            return TimeHorizon.INTRADAY
        if any(token in lowered for token in ("short-term", "this week", "week", "court terme")):
            return TimeHorizon.SHORT_TERM
        if any(token in lowered for token in ("swing", "multi-day")):
            return TimeHorizon.SWING
        return TimeHorizon.UNSPECIFIED

    def _extract_bias(self, text: str) -> tuple[UserBias, bool]:
        lowered = text.lower()
        if "force a buy" in lowered or "buy idea" in lowered:
            return UserBias.BUY_BIAS, True
        if "force a sell" in lowered or "sell idea" in lowered:
            return UserBias.SELL_BIAS, True
        if "prefer bullish" in lowered or "prefer a buy" in lowered:
            return UserBias.BUY_BIAS, False
        if "prefer bearish" in lowered or "prefer a sell" in lowered:
            return UserBias.SELL_BIAS, False
        return UserBias.NONE, False

    def _extract_mode(self, text: str, symbols: list[str]) -> RequestMode:
        lowered = text.lower()
        if symbols:
            return RequestMode.SINGLE_SYMBOL
        if any(
            phrase in lowered
            for phrase in (
                "best possible trades",
                "best trades",
                "what are the best",
                "scan the market",
                "opportunities this week",
                "find me a trade",
                "something safe",
                "try a trade",
            )
        ):
            return RequestMode.UNIVERSE_SCAN
        return RequestMode.SINGLE_SYMBOL

    def _operator_note_fr(
        self,
        risk: RiskPreference,
        horizon: TimeHorizon,
        bias: UserBias,
        refused: bool,
        mode: RequestMode,
    ) -> str:
        parts = [f"Mode de demande: {'scan univers' if mode == RequestMode.UNIVERSE_SCAN else 'symbole unique'}."]
        if risk == RiskPreference.CONSERVATIVE:
            parts.append("Le client demande une approche prudente.")
        elif risk == RiskPreference.AGGRESSIVE:
            parts.append("Le client accepte une posture plus agressive.")
        if horizon != TimeHorizon.UNSPECIFIED:
            parts.append(f"Horizon demandé: {horizon.value.lower()}.")
        if bias != UserBias.NONE:
            parts.append("Le client exprime un biais directionnel.")
        if refused:
            parts.append("Toute consigne de forçage directionnel doit être refusée.")
        return " ".join(parts)
