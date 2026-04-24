from __future__ import annotations

from typing import Any

from trading_agents.core.models import NewsChunk
from trading_agents.core.observability import LangSmithRetrievalTracer
from trading_agents.core.rag.store import BaseVectorStore


class NewsRetriever:
    def __init__(self, store: BaseVectorStore, tracer: LangSmithRetrievalTracer | None = None):
        self.store = store
        self.tracer = tracer

    def search_news(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[NewsChunk]:
        results = self.store.query("news", query, top_k=top_k) + self.store.query("macro_documents", query, top_k=top_k)
        deduped: dict[str, NewsChunk] = {}
        for chunk in results:
            deduped[chunk.chunk_id] = chunk
        ranked = sorted(deduped.values(), key=lambda chunk: chunk.similarity_score, reverse=True)
        selected = ranked[:top_k]
        if self.tracer is not None:
            self.tracer.log_search(
                query=query,
                top_k=top_k,
                filters=filters,
                collections=["news", "macro_documents"],
                results=selected,
                metadata=metadata,
            )
        return selected
