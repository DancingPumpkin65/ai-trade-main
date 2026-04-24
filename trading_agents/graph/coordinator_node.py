from __future__ import annotations

from trading_agents.core.models import CoordinatorOutput, IntentAlignment, RequestIntent, UserBias


def run_coordinator_agent(
    *,
    symbol: str,
    request_intent: RequestIntent,
    sentiment_output,
    technical_output,
    risk_output,
) -> CoordinatorOutput:
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

    rationale = (
        f"Demande comprise: {request_intent.operator_visible_note_fr} "
        f"Conclusion système sur {symbol}: {action}. "
        f"Le comité synthétise sentiment, technique et risque avant application des garde-fous."
    )
    if preference_conflicts:
        rationale += " " + " ".join(preference_conflicts)
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
