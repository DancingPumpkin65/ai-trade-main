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
