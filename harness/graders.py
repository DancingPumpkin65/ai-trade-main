from __future__ import annotations

from harness.models import GradeResult, HarnessScenario, ScenarioArtifact


def grade_scenario(
    scenario: HarnessScenario,
    *,
    initial_detail: dict,
    final_detail: dict,
    events: list[dict],
) -> GradeResult:
    failures: list[str] = []
    checks: list[str] = []
    intent = final_detail["request_intent"]
    expected_intent = scenario.expected_intent
    expected_outcome = scenario.expected_outcome

    if expected_intent.symbols_requested is not None:
        checks.append("intent.symbols_requested")
        if intent["symbols_requested"] != expected_intent.symbols_requested:
            failures.append(
                f"Expected symbols {expected_intent.symbols_requested}, got {intent['symbols_requested']}."
            )
    if expected_intent.request_mode is not None:
        checks.append("intent.request_mode")
        if intent["request_mode"] != expected_intent.request_mode:
            failures.append(f"Expected request_mode {expected_intent.request_mode}, got {intent['request_mode']}.")
    if expected_intent.risk_preference is not None:
        checks.append("intent.risk_preference")
        if intent["risk_preference"] != expected_intent.risk_preference:
            failures.append(
                f"Expected risk_preference {expected_intent.risk_preference}, got {intent['risk_preference']}."
            )
    if expected_intent.time_horizon is not None:
        checks.append("intent.time_horizon")
        if intent["time_horizon"] != expected_intent.time_horizon:
            failures.append(
                f"Expected time_horizon {expected_intent.time_horizon}, got {intent['time_horizon']}."
            )
    if expected_intent.user_bias is not None:
        checks.append("intent.user_bias")
        if intent["user_bias"] != expected_intent.user_bias:
            failures.append(f"Expected user_bias {expected_intent.user_bias}, got {intent['user_bias']}.")
    if expected_intent.bias_override_refused is not None:
        checks.append("intent.bias_override_refused")
        if intent["bias_override_refused"] != expected_intent.bias_override_refused:
            failures.append(
                "Expected bias_override_refused "
                f"{expected_intent.bias_override_refused}, got {intent['bias_override_refused']}."
            )

    if expected_outcome.initial_status is not None:
        checks.append("outcome.initial_status")
        if initial_detail["signal_status"] != expected_outcome.initial_status:
            failures.append(
                f"Expected initial status {expected_outcome.initial_status}, got {initial_detail['signal_status']}."
            )
    if expected_outcome.final_status is not None:
        checks.append("outcome.final_status")
        if final_detail["signal_status"] != expected_outcome.final_status:
            failures.append(f"Expected final status {expected_outcome.final_status}, got {final_detail['signal_status']}.")
    if expected_outcome.human_review_required_initial is not None:
        checks.append("outcome.human_review_required_initial")
        if initial_detail["human_review_required"] != expected_outcome.human_review_required_initial:
            failures.append(
                "Expected initial human_review_required "
                f"{expected_outcome.human_review_required_initial}, got {initial_detail['human_review_required']}."
            )
    if expected_outcome.final_action is not None:
        checks.append("outcome.final_action")
        action = (final_detail.get("final_signal") or {}).get("action")
        if action != expected_outcome.final_action:
            failures.append(f"Expected final action {expected_outcome.final_action}, got {action}.")
    if expected_outcome.intent_alignment is not None:
        checks.append("outcome.intent_alignment")
        coordinator = final_detail.get("coordinator_output") or {}
        alignment = coordinator.get("intent_alignment")
        if alignment != expected_outcome.intent_alignment:
            failures.append(f"Expected intent_alignment {expected_outcome.intent_alignment}, got {alignment}.")
    if expected_outcome.alpaca_order_status is not None:
        checks.append("outcome.alpaca_order_status")
        if final_detail["alpaca_order_status"] != expected_outcome.alpaca_order_status:
            failures.append(
                "Expected alpaca_order_status "
                f"{expected_outcome.alpaca_order_status}, got {final_detail['alpaca_order_status']}."
            )
    if expected_outcome.opportunity_count is not None:
        checks.append("outcome.opportunity_count")
        opportunity_count = len((final_detail.get("opportunity_list") or {}).get("top_opportunities", []))
        if opportunity_count != expected_outcome.opportunity_count:
            failures.append(f"Expected {expected_outcome.opportunity_count} opportunities, got {opportunity_count}.")
    if expected_outcome.top_symbols is not None:
        checks.append("outcome.top_symbols")
        opportunities = (final_detail.get("opportunity_list") or {}).get("top_opportunities", [])
        actual_symbols = [item["signal"]["symbol"] for item in opportunities]
        missing = [symbol for symbol in expected_outcome.top_symbols if symbol not in actual_symbols]
        if missing:
            failures.append(f"Expected top symbols {missing} to appear in opportunities, got {actual_symbols}.")
    if expected_outcome.min_event_count is not None:
        checks.append("outcome.min_event_count")
        if len(events) < expected_outcome.min_event_count:
            failures.append(f"Expected at least {expected_outcome.min_event_count} events, got {len(events)}.")

    if scenario.expected_event_types:
        checks.append("events.expected_types")
        actual_types = [event["event_type"] for event in events]
        missing = [event_type for event_type in scenario.expected_event_types if event_type not in actual_types]
        if missing:
            failures.append(f"Missing expected event types: {missing}.")

    return GradeResult(passed=not failures, failures=failures, checks=checks)


def replay_artifact(artifact: ScenarioArtifact) -> GradeResult:
    return grade_scenario(
        artifact.scenario,
        initial_detail=artifact.initial_detail,
        final_detail=artifact.final_detail,
        events=artifact.events,
    )
