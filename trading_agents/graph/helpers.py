from __future__ import annotations

from math import sqrt

from trading_agents.core.models import MarketMode, PositionSizing, StockInfo, TechnicalFeatures


def _ema(values: list[float], period: int) -> float:
    multiplier = 2 / (period + 1)
    ema = values[0]
    for value in values[1:]:
        ema = (value - ema) * multiplier + ema
    return ema


def detect_market_mode(stock: StockInfo) -> MarketMode:
    if stock.market_mode != MarketMode.UNKNOWN:
        return stock.market_mode
    haystack = " ".join(
        [
            stock.name,
            stock.sector,
            str(stock.market_metadata.get("market_mode", "")),
            str(stock.market_metadata.get("trading_mode", "")),
            str(stock.market_metadata.get("quote_mode", "")),
            str(stock.market_metadata.get("instrument_type", "")),
            str(stock.market_metadata.get("asset_type", "")),
            str(stock.market_metadata.get("compartment", "")),
        ]
    ).lower()
    if any(token in haystack for token in ("bond", "obligation", "fixed income", "treasury", "debt")):
        return MarketMode.BOND
    if any(token in haystack for token in ("fixing", "fixe")):
        return MarketMode.FIXING
    if any(token in haystack for token in ("continuous", "continu")):
        return MarketMode.CONTINUOUS
    if stock.last_price < 80 and stock.last_volume < 250_000:
        return MarketMode.FIXING
    return MarketMode.CONTINUOUS


def normalize_market_mode(mode: MarketMode | str | None) -> MarketMode:
    if isinstance(mode, MarketMode):
        return mode
    if isinstance(mode, str):
        try:
            return MarketMode(mode)
        except ValueError:
            return MarketMode.UNKNOWN
    return MarketMode.UNKNOWN


def is_fixing_market(mode: MarketMode | str | None) -> bool:
    return normalize_market_mode(mode) == MarketMode.FIXING


def market_mode_daily_limit(mode: MarketMode | str | None) -> float:
    resolved = normalize_market_mode(mode)
    if resolved == MarketMode.BOND:
        return 0.02
    if resolved == MarketMode.FIXING:
        return 0.06
    return 0.10


def market_mode_static_reservation_limit(mode: MarketMode | str | None) -> float | None:
    resolved = normalize_market_mode(mode)
    if resolved == MarketMode.FIXING:
        return 0.04
    if resolved == MarketMode.CONTINUOUS:
        return 0.06
    return None


def market_mode_dynamic_reservation_limit(mode: MarketMode | str | None) -> float | None:
    resolved = normalize_market_mode(mode)
    if resolved == MarketMode.BOND:
        return None
    return 0.03


def analyze_technical_features(stock: StockInfo) -> TechnicalFeatures:
    closes = [bar["close"] for bar in stock.ohlcv if bar.get("close") is not None]
    highs = [bar["high"] for bar in stock.ohlcv if bar.get("high") is not None]
    lows = [bar["low"] for bar in stock.ohlcv if bar.get("low") is not None]
    volumes = [bar["volume"] for bar in stock.ohlcv if bar.get("volume") is not None]
    if len(closes) < 20:
        closes = closes + [stock.last_price] * max(0, 20 - len(closes))
        highs = highs + [stock.last_price] * max(0, 20 - len(highs))
        lows = lows + [stock.last_price] * max(0, 20 - len(lows))
        volumes = volumes + [stock.last_volume] * max(0, 20 - len(volumes))

    sma20 = sum(closes[-20:]) / 20
    ema10 = _ema(closes[-10:], 10)
    gains = []
    losses = []
    for prev, current in zip(closes[-15:-1], closes[-14:]):
        delta = current - prev
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))
    avg_gain = sum(gains) / 14 if gains else 0.0
    avg_loss = sum(losses) / 14 if losses else 0.0
    rs = avg_gain / avg_loss if avg_loss else 100.0
    rsi14 = 100 - (100 / (1 + rs))
    true_ranges = [high - low for high, low in zip(highs[-14:], lows[-14:])]
    atr14 = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    mean = sma20
    variance = sum((close - mean) ** 2 for close in closes[-20:]) / 20
    std_dev = sqrt(variance)
    upper = mean + 2 * std_dev
    lower = mean - 2 * std_dev
    width = (upper - lower) / mean if mean else 0.0

    returns = []
    for prev, current in zip(closes[:-1], closes[1:]):
        if prev:
            returns.append((current - prev) / prev)
    avg_return = sum(returns) / len(returns) if returns else 0.0
    vol = sqrt(sum((item - avg_return) ** 2 for item in returns) / len(returns)) * sqrt(252) if returns else 0.0
    support_levels = sorted({round(value, 2) for value in lows[-5:]})[:3]
    resistance_levels = sorted({round(value, 2) for value in highs[-5:]}, reverse=True)[:3]
    directional_bias = "BULLISH" if ema10 > sma20 else "BEARISH" if ema10 < sma20 else "NEUTRAL"
    market_mode = detect_market_mode(stock)

    return TechnicalFeatures(
        sma20=round(sma20, 4),
        ema10=round(ema10, 4),
        rsi14=round(rsi14, 4),
        atr14=round(atr14, 4),
        bollinger_upper=round(upper, 4),
        bollinger_lower=round(lower, 4),
        bollinger_width=round(width, 4),
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        directional_bias=directional_bias,
        annualized_volatility=round(abs(vol), 4),
        zero_volume_bar_count=sum(1 for volume in volumes if volume == 0),
        market_mode=market_mode,
        is_fixing_mode=is_fixing_market(market_mode),
    )


def calculate_position_size(
    symbol: str,
    action: str,
    capital: float,
    volatility_estimate: float,
    is_fixing_mode: bool,
    market_mode: MarketMode = MarketMode.UNKNOWN,
    conservative_posture: bool = False,
) -> PositionSizing:
    resolved_market_mode = MarketMode.FIXING if is_fixing_mode else normalize_market_mode(market_mode)
    daily_limit = market_mode_daily_limit(resolved_market_mode)
    base_size = 0.03 if conservative_posture else 0.05
    position_size_pct = min(base_size, max(0.01, 0.05 - volatility_estimate * 0.03))
    if is_fixing_mode:
        position_size_pct *= 0.8
    stop_loss_pct = min(daily_limit * 0.8, max(0.02, volatility_estimate * 0.4))
    take_profit_pct = max(stop_loss_pct * 1.5, 0.03)
    risk_score = min(1.0, volatility_estimate + (0.15 if action in {"BUY", "SELL"} else 0.05))
    return PositionSizing(
        position_size_pct=round(min(position_size_pct, 0.05), 4),
        position_value_mad=round(capital * min(position_size_pct, 0.05), 2),
        stop_loss_pct=round(min(stop_loss_pct, daily_limit), 4),
        take_profit_pct=round(max(take_profit_pct, stop_loss_pct * 1.5), 4),
        risk_score=round(risk_score, 4),
        volatility_estimate=round(volatility_estimate, 4),
        market_mode=resolved_market_mode,
        is_fixing_mode=is_fixing_mode,
    )
