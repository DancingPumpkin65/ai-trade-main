export type RiskPreference = "CONSERVATIVE" | "BALANCED" | "AGGRESSIVE";
export type TimeHorizon = "INTRADAY" | "SHORT_TERM" | "SWING" | "UNSPECIFIED";
export type UserBias = "NONE" | "BUY_BIAS" | "SELL_BIAS";
export type RequestMode = "SINGLE_SYMBOL" | "UNIVERSE_SCAN";
export type IntentAlignment = "ALIGNED" | "PARTIALLY_ALIGNED" | "NOT_ALIGNED";
export type SignalStatus =
  | "PENDING"
  | "RUNNING"
  | "WAITING_HUMAN"
  | "APPROVED"
  | "REJECTED"
  | "COMPLETED"
  | "FAILED";
export type AlpacaOrderStatus = "NOT_PREPARED" | "PREPARED" | "APPROVED" | "REJECTED" | "UNMAPPABLE";
export type MarketMode = "CONTINUOUS" | "FIXING" | "BOND" | "UNKNOWN";

export interface RequestIntent {
  request_id: string;
  raw_prompt: string | null;
  symbols_requested: string[];
  capital_mad: number;
  request_mode: RequestMode;
  risk_preference: RiskPreference;
  time_horizon: TimeHorizon;
  user_bias: UserBias;
  bias_override_refused: boolean;
  intent_notes_en: string;
  operator_visible_note_fr: string;
  parser_confidence: number;
  extraction_method: string;
}

export interface TradingSignal {
  symbol: string;
  action: string;
  position_size_pct: number;
  position_value_mad: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  risk_score: number;
  gap_risk_warning: string | null;
  execution_warnings: string[];
  rationale_fr: string;
  confidence: number;
  market_mode: MarketMode;
  is_fixing_mode: boolean;
  request_id: string;
  generated_at: string;
}

export interface CoordinatorOutput {
  action: string;
  position_size_pct: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  risk_score: number;
  rationale_fr: string;
  dissenting_views: string[];
  confidence: number;
  intent_alignment: IntentAlignment;
  preference_conflicts: string[];
}

export interface TradeOpportunity {
  rank: number;
  signal: TradingSignal;
  coordinator_output: CoordinatorOutput;
  intent_alignment: IntentAlignment;
}

export interface TradeOpportunityList {
  request_id: string;
  capital_mad: number;
  time_horizon: TimeHorizon;
  risk_preference: RiskPreference;
  top_opportunities: TradeOpportunity[];
  rejected_candidates_summary: string[];
}

export interface AlpacaOrderIntent {
  request_id: string;
  client_order_id: string;
  source_symbol: string;
  alpaca_symbol: string | null;
  side: string | null;
  type: string | null;
  time_in_force: string | null;
  qty: number | null;
  notional: number | null;
  preview_only: boolean;
  submission_eligible: boolean;
  status: AlpacaOrderStatus;
  reason: string | null;
  broker_order_id?: string | null;
  broker_order_status?: string | null;
  broker_submission_mode?: string | null;
  submitted_at?: string | null;
  created_at: string;
}

export interface SignalRecord {
  request_id: string;
  status: SignalStatus;
  request_intent: RequestIntent;
  final_signal: TradingSignal | null;
  opportunity_list: TradeOpportunityList | null;
  coordinator_output: CoordinatorOutput | null;
  human_review_required: boolean;
  alpaca_order_status: AlpacaOrderStatus;
  alpaca_order: AlpacaOrderIntent | null;
  errors: string[];
}

export interface UniverseScanCandidateRecord {
  request_id: string;
  symbol: string;
  score: number | null;
  reasons: string[];
  selected_for_deep_eval: boolean;
  rank_position: number | null;
  evaluation_status: string;
  rejection_reason: string | null;
}

export interface SignalDetailResponse extends SignalRecord {
  signal_status: SignalStatus;
  order_approval_required: boolean;
  analysis_warnings: string[];
  universe_scan_candidates: UniverseScanCandidateRecord[];
  opportunity_alpaca_orders: Record<string, AlpacaOrderIntent>;
}

export interface GenerateSignalRequest {
  symbol?: string;
  capital?: number;
  prompt?: string;
  risk_profile?: RiskPreference;
  time_horizon?: TimeHorizon;
}

export interface GenerateSignalResponse {
  request_id: string;
  status: SignalStatus;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface HealthResponse {
  status: string;
  db_path: string;
  langgraph_checkpoint_path: string;
  langsmith_tracing: boolean;
  alpaca_enabled: boolean;
  alpaca_require_order_approval: boolean;
  alpaca_submit_orders: boolean;
  alpaca_submission_mode: string;
  langgraph_enabled: boolean;
  rag_backend: string;
  bourse_cache_dir: string;
}

export interface SignalEvent {
  event_type: string;
  payload: Record<string, unknown>;
}

export interface AlpacaOrderEnvelope {
  request_id: string;
  alpaca_order_status: AlpacaOrderStatus;
  order_approval_required: boolean;
  alpaca_order: AlpacaOrderIntent | null;
}

export interface OpportunityAlpacaOrderEnvelope extends AlpacaOrderEnvelope {
  symbol: string;
}
