from __future__ import annotations

from trading_agents.core.models import RequestIntent, SentimentOutput


def run_sentiment_agent(
    *,
    symbol: str,
    request_intent: RequestIntent,
    market_data,
    news_chunks,
) -> SentimentOutput:
    catalysts: list[str] = []
    article_ids: list[str] = []
    positive_hits = 0
    negative_hits = 0
    for chunk in news_chunks[:5]:
        text = chunk.text.lower()
        if any(token in text for token in ("dividend", "earnings", "growth", "contract", "hausse")):
            positive_hits += 1
            catalysts.append(f"Catalyseur identifié via {chunk.source}: {chunk.text[:90].strip()}")
            article_ids.append(chunk.chunk_id)
        elif any(token in text for token in ("warning", "downgrade", "lawsuit", "baisse", "decline")):
            negative_hits += 1
            catalysts.append(f"Risque identifié via {chunk.source}: {chunk.text[:90].strip()}")
            article_ids.append(chunk.chunk_id)
    score = 0.5
    if positive_hits > negative_hits:
        score = min(1.0, 0.55 + positive_hits * 0.1)
    elif negative_hits > positive_hits:
        score = max(0.0, 0.45 - negative_hits * 0.1)
    confidence = min(0.9, 0.35 + len(article_ids) * 0.1)
    if len(catalysts) == 0:
        confidence = min(confidence, 0.30)
    elif len(catalysts) == 1:
        confidence = min(confidence, 0.45)

    intent_clause = ""
    if request_intent.bias_override_refused:
        intent_clause = " Le souhait directionnel exprimé par le client a été noté, mais il n'est pas traité comme une preuve de marché."
    rationale_fr = (
        f"Analyse de sentiment pour {symbol}: {len(article_ids)} sources pertinentes ont été retenues. "
        f"Le ton global est {'positif' if score > 0.55 else 'négatif' if score < 0.45 else 'mitigé'}."
        f"{intent_clause}"
    )
    return SentimentOutput(
        sentiment_score=round(score, 4),
        catalysts=catalysts[:5],
        cited_article_ids=article_ids[:5],
        confidence=round(confidence, 4),
        rationale_fr=rationale_fr,
    )
