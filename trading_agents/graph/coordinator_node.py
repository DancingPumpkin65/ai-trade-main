from __future__ import annotations

from trading_agents.core.intent.policy import IntentPolicyEngine
from trading_agents.core.llm import get_default_agent_llm
from trading_agents.core.models import CoordinatorOutput, IntentAlignment, RequestIntent, UserBias


def _deterministic_coordinator_output(
    *,
    symbol: str,
    request_intent: RequestIntent,
    sentiment_output,
    technical_output,
    risk_output,
    policy_context: dict | None = None,
) -> CoordinatorOutput:
    policy_engine = IntentPolicyEngine()
    policy = policy_engine.build(request_intent)
    context = policy_context or policy_engine.build_coordinator_prompt_context(request_intent)
    dissenting_views: list[str] = []
    preference_conflicts: list[str] = []
    if sentiment_output.sentiment_score >= 0.5 and technical_output.directional_bias == "BEARISH":
        dissenting_views.append("Le sentiment est plus favorable que le signal technique.")
    if sentiment_output.sentiment_score < 0.5 and technical_output.directional_bias == "BULLISH":
        dissenting_views.append("Le signal technique est plus fort que le flux d'actualités.")

    action = risk_output.action
    alignment = IntentAlignment.ALIGNED
    if request_intent.user_bias == UserBias.BUY_BIAS and action != "BUY":
        alignment = IntentAlignment.NOT_ALIGNED
        preference_conflicts.append("La préférence haussière du client n'est pas retenue au vu des données.")
    elif request_intent.user_bias == UserBias.SELL_BIAS and action != "SELL":
        alignment = IntentAlignment.NOT_ALIGNED
        preference_conflicts.append("La préférence baissière du client n'est pas retenue au vu des données.")
    elif request_intent.risk_preference.value == "CONSERVATIVE" and action in {"BUY", "SELL"}:
        alignment = IntentAlignment.PARTIALLY_ALIGNED
        preference_conflicts.append("Le signal reste opportuniste mais la taille est réduite pour respecter la prudence demandée.")

    sentiment_tone = (
        "sentiment favorable"
        if sentiment_output.sentiment_score >= 0.55
        else "sentiment prudent"
        if sentiment_output.sentiment_score <= 0.45
        else "sentiment mitige"
    )
    technical_tone = {
        "BULLISH": "biais technique haussier",
        "BEARISH": "biais technique baissier",
    }.get(technical_output.directional_bias, "biais technique neutre")
    sections = [
        f"{context['request_frame']}: {request_intent.operator_visible_note_fr}",
        f"{context['interpretation_frame']}: {context['interpretation_focus']} Le dossier combine {sentiment_tone} et {technical_tone}.",
        f"{context['recommendation_frame']} sur {symbol}: {action}.",
        f"{context['synthesis_frame']}: {policy.coordinator_note}",
    ]
    if dissenting_views:
        sections.append(f"{context['dissent_frame']}: {' '.join(dissenting_views)}")
    if preference_conflicts:
        sections.append(f"{context['conflict_frame']}: {' '.join(preference_conflicts)}")
    rationale = " ".join(section.strip() for section in sections if section.strip())
    return CoordinatorOutput(
        action=action,
        position_size_pct=risk_output.position_size_pct,
        stop_loss_pct=risk_output.stop_loss_pct,
        take_profit_pct=risk_output.take_profit_pct,
        risk_score=risk_output.risk_score,
        rationale_fr=rationale,
        dissenting_views=dissenting_views,
        confidence=round((sentiment_output.confidence + technical_output.confidence) / 2, 4),
        intent_alignment=alignment,
        preference_conflicts=preference_conflicts,
    )


def run_coordinator_agent(
    *,
    symbol: str,
    request_intent: RequestIntent,
    sentiment_output,
    technical_output,
    risk_output,
    policy_context: dict | None = None,
) -> CoordinatorOutput:
    deterministic = _deterministic_coordinator_output(
        symbol=symbol,
        request_intent=request_intent,
        sentiment_output=sentiment_output,
        technical_output=technical_output,
        risk_output=risk_output,
        policy_context=policy_context,
    )
    llm = get_default_agent_llm()
    if llm is None:
        return deterministic

    llm_output = llm.generate_structured(
        agent_name="coordinator",
        system_prompt=(
            "Produce the final French coordinator summary for a Morocco trading workflow. "
            "Respect the supplied risk action as the final action. "
            "Clearly distinguish user preference from system conclusion when they diverge."
        ),
        context={
            "symbol": symbol,
            "request_intent": request_intent.model_dump(mode="json"),
            "policy_context": policy_context,
            "sentiment_output": sentiment_output.model_dump(mode="json"),
            "technical_output": technical_output.model_dump(mode="json"),
            "risk_output": risk_output.model_dump(mode="json"),
            "deterministic_baseline": deterministic.model_dump(mode="json"),
        },
        response_model=CoordinatorOutput,
    )
    if llm_output is None:
        return deterministic

    return deterministic.model_copy(
        update={
            "rationale_fr": llm_output.rationale_fr or deterministic.rationale_fr,
            "dissenting_views": llm_output.dissenting_views or deterministic.dissenting_views,
            "confidence": round(max(0.0, min(1.0, llm_output.confidence)), 4),
            "intent_alignment": llm_output.intent_alignment,
            "preference_conflicts": llm_output.preference_conflicts or deterministic.preference_conflicts,
        }
    )
