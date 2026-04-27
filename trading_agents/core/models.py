from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RequestMode(str, Enum):
    SINGLE_SYMBOL = "SINGLE_SYMBOL"
    UNIVERSE_SCAN = "UNIVERSE_SCAN"


class RiskPreference(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"


class TimeHorizon(str, Enum):
    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"
    SWING = "SWING"
    UNSPECIFIED = "UNSPECIFIED"


class UserBias(str, Enum):
    NONE = "NONE"
    BUY_BIAS = "BUY_BIAS"
    SELL_BIAS = "SELL_BIAS"


class IntentAlignment(str, Enum):
    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    NOT_ALIGNED = "NOT_ALIGNED"


class SignalStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AlpacaOrderStatus(str, Enum):
    NOT_PREPARED = "NOT_PREPARED"
    PREPARED = "PREPARED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNMAPPABLE = "UNMAPPABLE"


class MarketMode(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    FIXING = "FIXING"
    BOND = "BOND"
    UNKNOWN = "UNKNOWN"


class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    market_cap: float = 0.0
    last_price: float = 0.0
    last_volume: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    market_mode: MarketMode = MarketMode.UNKNOWN
    market_metadata: dict = Field(default_factory=dict)
    ohlcv: list[dict] = Field(default_factory=list)


class NewsChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    published_at: datetime | None = None
    similarity_score: float = 0.0
    url: str | None = None
    low_confidence: bool = False
    metadata: dict = Field(default_factory=dict)


class TechnicalFeatures(BaseModel):
    sma20: float
    ema10: float
    rsi14: float
    atr14: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_width: float
    support_levels: list[float]
    resistance_levels: list[float]
    directional_bias: str
    annualized_volatility: float
    zero_volume_bar_count: int
    market_mode: MarketMode = MarketMode.UNKNOWN
    is_fixing_mode: bool = False


class PositionSizing(BaseModel):
    position_size_pct: float
    position_value_mad: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_score: float
    volatility_estimate: float
    market_mode: MarketMode = MarketMode.UNKNOWN
    is_fixing_mode: bool = False


class RequestIntent(BaseModel):
    request_id: str
    raw_prompt: str | None = None
    symbols_requested: list[str] = Field(default_factory=list)
    capital_mad: float = 0.0
    request_mode: RequestMode = RequestMode.SINGLE_SYMBOL
    risk_preference: RiskPreference = RiskPreference.BALANCED
    time_horizon: TimeHorizon = TimeHorizon.UNSPECIFIED
    user_bias: UserBias = UserBias.NONE
    bias_override_refused: bool = False
    intent_notes_en: str = ""
    operator_visible_note_fr: str = ""
    parser_confidence: float = 1.0
    extraction_method: str = "deterministic"


class SentimentOutput(BaseModel):
    sentiment_score: float
    catalysts: list[str]
    cited_article_ids: list[str]
    confidence: float
    rationale_fr: str


class TechnicalOutput(BaseModel):
    directional_bias: str
    trend_summary: str
    momentum_summary: str
    volatility_summary: str
    support_levels: list[float]
    resistance_levels: list[float]
    volatility_estimate: float
    liquidity_comment: str
    confidence: float


class RiskOutput(BaseModel):
    action: str
    position_size_pct: float
    position_value_mad: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_score: float
    volatility_estimate: float
    rationale: str


class CoordinatorOutput(BaseModel):
    action: str
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_score: float
    rationale_fr: str
    dissenting_views: list[str]
    confidence: float
    intent_alignment: IntentAlignment = IntentAlignment.ALIGNED
    preference_conflicts: list[str] = Field(default_factory=list)


class TradingSignal(BaseModel):
    symbol: str
    action: str
    position_size_pct: float
    position_value_mad: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_score: float
    gap_risk_warning: str | None = None
    execution_warnings: list[str] = Field(default_factory=list)
    rationale_fr: str
    confidence: float
    market_mode: MarketMode = MarketMode.UNKNOWN
    is_fixing_mode: bool
    request_id: str
    generated_at: datetime


class RankedCandidate(BaseModel):
    symbol: str
    score: float
    reasons: list[str]


class UniverseScanCandidateRecord(BaseModel):
    request_id: str
    symbol: str
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    selected_for_deep_eval: bool = False
    rank_position: int | None = None
    evaluation_status: str
    rejection_reason: str | None = None


class TradeOpportunity(BaseModel):
    rank: int
    signal: TradingSignal
    coordinator_output: CoordinatorOutput
    intent_alignment: IntentAlignment


class TradeOpportunityList(BaseModel):
    request_id: str
    capital_mad: float
    time_horizon: TimeHorizon
    risk_preference: RiskPreference
    top_opportunities: list[TradeOpportunity]
    rejected_candidates_summary: list[str]


class AlpacaOrderIntent(BaseModel):
    request_id: str
    client_order_id: str
    source_symbol: str
    alpaca_symbol: str | None
    side: str | None
    type: str | None
    time_in_force: str | None
    qty: float | None = None
    notional: float | None = None
    preview_only: bool = True
    submission_eligible: bool = False
    status: AlpacaOrderStatus = AlpacaOrderStatus.NOT_PREPARED
    reason: str | None = None
    created_at: datetime


class GenerateSignalRequest(BaseModel):
    symbol: str | None = None
    capital: float | None = None
    prompt: str | None = None
    risk_profile: RiskPreference | None = None
    time_horizon: TimeHorizon | None = None


class GenerateSignalResponse(BaseModel):
    request_id: str
    status: SignalStatus


class SignalRecord(BaseModel):
    request_id: str
    status: SignalStatus
    request_intent: RequestIntent
    final_signal: TradingSignal | None = None
    opportunity_list: TradeOpportunityList | None = None
    coordinator_output: CoordinatorOutput | None = None
    human_review_required: bool = False
    alpaca_order_status: AlpacaOrderStatus = AlpacaOrderStatus.NOT_PREPARED
    alpaca_order: AlpacaOrderIntent | None = None
    errors: list[str] = Field(default_factory=list)


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str
