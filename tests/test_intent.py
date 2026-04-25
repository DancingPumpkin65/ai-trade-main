from trading_agents.core.intent.normalizer import NormalizedIntentHints
from trading_agents.core.intent.parser import IntentParser
from trading_agents.core.models import GenerateSignalRequest, RequestMode, RiskPreference, TimeHorizon, UserBias


def test_parse_symbol_capital_and_conservative_prompt():
    parser = IntentParser()
    intent = parser.parse(GenerateSignalRequest(prompt="Analyze ATW with conservative risk and 100,000 MAD capital"))
    assert intent.symbols_requested == ["ATW"]
    assert intent.capital_mad == 100000
    assert intent.risk_preference == RiskPreference.CONSERVATIVE


def test_parse_universe_scan_short_term():
    parser = IntentParser()
    intent = parser.parse(GenerateSignalRequest(prompt="I have 100,000 MAD. What are the best possible trades this week?"))
    assert intent.request_mode == RequestMode.UNIVERSE_SCAN
    assert intent.time_horizon == TimeHorizon.SHORT_TERM


def test_force_buy_is_refused():
    parser = IntentParser()
    intent = parser.parse(GenerateSignalRequest(prompt="Analyze ATW and force a BUY idea"))
    assert intent.user_bias == UserBias.BUY_BIAS
    assert intent.bias_override_refused is True


def test_ambiguous_prompt_uses_llm_fallback_for_universe_scan():
    class FakeNormalizer:
        def normalize(self, prompt: str):
            assert prompt == "Try a short-term trade"
            return NormalizedIntentHints(
                request_mode=RequestMode.UNIVERSE_SCAN,
                time_horizon=TimeHorizon.SHORT_TERM,
                risk_preference=RiskPreference.BALANCED,
                parser_confidence=0.84,
                intent_notes_en="Ambiguous prompt interpreted as a universe scan request.",
                ambiguity_reason="No explicit symbol was provided.",
            )

    parser = IntentParser(normalizer=FakeNormalizer())
    intent = parser.parse(GenerateSignalRequest(prompt="Try a short-term trade"))
    assert intent.request_mode == RequestMode.UNIVERSE_SCAN
    assert intent.time_horizon == TimeHorizon.SHORT_TERM
    assert intent.parser_confidence == 0.84
    assert intent.extraction_method == "deterministic+llm"


def test_llm_fallback_can_supply_missing_capital_and_symbol():
    class FakeNormalizer:
        def normalize(self, prompt: str):
            return NormalizedIntentHints(
                symbols_requested=["ATW"],
                capital_mad=50000,
                request_mode=RequestMode.SINGLE_SYMBOL,
                risk_preference=RiskPreference.CONSERVATIVE,
                parser_confidence=0.88,
                intent_notes_en="Mapped the prompt to ATW with conservative posture.",
            )

    parser = IntentParser(normalizer=FakeNormalizer())
    intent = parser.parse(GenerateSignalRequest(prompt="Give me something safe with fifty thousand dirhams"))
    assert intent.symbols_requested == ["ATW"]
    assert intent.capital_mad == 50000
    assert intent.risk_preference == RiskPreference.CONSERVATIVE
    assert intent.extraction_method == "deterministic+llm"


def test_deterministic_parse_does_not_call_llm_when_prompt_is_clear():
    class FailingNormalizer:
        def normalize(self, prompt: str):
            raise AssertionError("LLM fallback should not be used for clear prompts.")

    parser = IntentParser(normalizer=FailingNormalizer())
    intent = parser.parse(GenerateSignalRequest(prompt="Analyze ATW with conservative risk"))
    assert intent.symbols_requested == ["ATW"]
    assert intent.extraction_method == "deterministic"


def test_intent_tracer_receives_normalization_payload():
    class FakeNormalizer:
        def normalize(self, prompt: str):
            return NormalizedIntentHints(
                request_mode=RequestMode.UNIVERSE_SCAN,
                parser_confidence=0.75,
                intent_notes_en="Resolved to a universe scan.",
            )

    class FakeTracer:
        def __init__(self):
            self.calls = []

        def log_normalization(self, **kwargs):
            self.calls.append(kwargs)

    tracer = FakeTracer()
    parser = IntentParser(normalizer=FakeNormalizer(), tracer=tracer)
    intent = parser.parse(GenerateSignalRequest(prompt="Try a trade"))
    assert intent.request_mode == RequestMode.UNIVERSE_SCAN
    assert len(tracer.calls) == 1
    assert tracer.calls[0]["prompt"] == "Try a trade"
    assert tracer.calls[0]["normalized"]["request_mode"] == "UNIVERSE_SCAN"
