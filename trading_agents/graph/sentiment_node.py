from __future__ import annotations

from trading_agents.core.llm import get_default_agent_llm
from trading_agents.core.models import RequestIntent, SentimentOutput


def _deterministic_sentiment_agent(
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


def _post_process_sentiment_output(
    output: SentimentOutput,
    *,
    request_intent: RequestIntent,
    available_chunk_ids: set[str],
) -> SentimentOutput:
    cited_article_ids = [chunk_id for chunk_id in output.cited_article_ids if chunk_id in available_chunk_ids][:5]
    catalysts = output.catalysts[:5]
    confidence = max(0.0, min(1.0, output.confidence))
    if len(catalysts) == 0:
        confidence = min(confidence, 0.30)
    elif len(catalysts) == 1:
        confidence = min(confidence, 0.45)
    if request_intent.bias_override_refused and "préférence" not in output.rationale_fr.lower():
        rationale = (
            f"{output.rationale_fr} La préférence directionnelle du client est notée mais non utilisée comme preuve."
        )
    else:
        rationale = output.rationale_fr
    return output.model_copy(
        update={
            "sentiment_score": round(max(0.0, min(1.0, output.sentiment_score)), 4),
            "catalysts": catalysts,
            "cited_article_ids": cited_article_ids,
            "confidence": round(confidence, 4),
            "rationale_fr": rationale,
        }
    )


def run_sentiment_agent(
    *,
    symbol: str,
    request_intent: RequestIntent,
    market_data,
    news_chunks,
) -> SentimentOutput:
    deterministic = _deterministic_sentiment_agent(
        symbol=symbol,
        request_intent=request_intent,
        market_data=market_data,
        news_chunks=news_chunks,
    )
    llm = get_default_agent_llm()
    if llm is None:
        return deterministic

    llm_output = llm.generate_structured(
        agent_name="sentiment",
        system_prompt=(
            "Assess sentiment for a Moroccan equity using only the provided market/news context. "
            "Do not invent sources. "
            "Use cited_article_ids only from the provided chunk ids. "
            "Treat user directional preference as context, never evidence."
        ),
        context={
            "symbol": symbol,
            "request_intent": request_intent.model_dump(mode="json"),
            "market_data": {
                "name": getattr(market_data, "name", symbol),
                "sector": getattr(market_data, "sector", "Unknown"),
                "last_price": getattr(market_data, "last_price", 0.0),
                "last_volume": getattr(market_data, "last_volume", 0.0),
            },
            "news_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text": chunk.text[:600],
                    "published_at": chunk.published_at.isoformat() if chunk.published_at else None,
                    "metadata": chunk.metadata,
                }
                for chunk in news_chunks[:8]
            ],
            "deterministic_baseline": deterministic.model_dump(mode="json"),
        },
        response_model=SentimentOutput,
    )
    if llm_output is None:
        return deterministic
    return _post_process_sentiment_output(
        llm_output,
        request_intent=request_intent,
        available_chunk_ids={chunk.chunk_id for chunk in news_chunks},
    )
