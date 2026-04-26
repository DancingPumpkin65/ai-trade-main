from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from trading_agents.core.models import GenerateSignalRequest


class ScenarioExpectedIntent(BaseModel):
    symbols_requested: list[str] | None = None
    request_mode: str | None = None
    risk_preference: str | None = None
    time_horizon: str | None = None
    user_bias: str | None = None
    bias_override_refused: bool | None = None


class ScenarioExpectedOutcome(BaseModel):
    initial_status: str | None = None
    final_status: str | None = None
    human_review_required_initial: bool | None = None
    final_action: str | None = None
    intent_alignment: str | None = None
    alpaca_order_status: str | None = None
    opportunity_count: int | None = None
    top_symbols: list[str] | None = None
    min_event_count: int | None = None


class HarnessScenario(BaseModel):
    name: str
    description: str
    fixture: str
    request: GenerateSignalRequest
    approval_decision: str = "none"
    expected_intent: ScenarioExpectedIntent = Field(default_factory=ScenarioExpectedIntent)
    expected_outcome: ScenarioExpectedOutcome = Field(default_factory=ScenarioExpectedOutcome)
    expected_event_types: list[str] = Field(default_factory=list)


class GradeResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class ScenarioArtifact(BaseModel):
    name: str
    description: str
    fixture: str
    request_id: str
    approval_decision: str
    scenario: HarnessScenario
    initial_detail: dict[str, Any]
    final_detail: dict[str, Any]
    events: list[dict[str, Any]]
    generated_at: datetime
    grade: GradeResult


class HarnessReport(BaseModel):
    mode: str
    generated_at: datetime
    scenario_count: int
    passed_count: int
    failed_count: int
    artifacts: list[ScenarioArtifact]
