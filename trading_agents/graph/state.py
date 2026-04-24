from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from trading_agents.core.models import (
    CoordinatorOutput,
    NewsChunk,
    RequestIntent,
    RiskOutput,
    SentimentOutput,
    StockInfo,
    TechnicalOutput,
    TradingSignal,
)


class GraphState(TypedDict):
    symbol: str
    capital: float
    request_id: str
    request_intent: RequestIntent
    candidate_symbols: list[str]
    selected_symbol: str | None
    intent_warnings: list[str]
    stock_info: StockInfo | None
    news_chunks: list[NewsChunk]
    technical_features: dict | None
    sentiment_output: SentimentOutput | None
    technical_output: TechnicalOutput | None
    risk_output: RiskOutput | None
    coordinator_output: CoordinatorOutput | None
    final_signal: TradingSignal | None
    sentiment_messages: list[dict]
    technical_messages: list[dict]
    risk_messages: list[dict]
    coordinator_messages: list[dict]
    technical_retry_count: int
    human_review_required: bool
    human_review_decision: Literal["approved", "rejected"] | None
    started_at: datetime
    errors: list[str]
