from __future__ import annotations

from trading_agents.core.models import NewsChunk
from trading_agents.core.rag.store import InMemoryVectorStore


class NewsRetriever:
    def __init__(self, store: InMemoryVectorStore):
        self.store = store

    def search_news(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[NewsChunk]:
        results = self.store.query("news", query, top_k=top_k) + self.store.query("macro_documents", query, top_k=top_k)
        deduped: dict[str, NewsChunk] = {}
        for chunk in results:
            deduped[chunk.chunk_id] = chunk
        ranked = sorted(deduped.values(), key=lambda chunk: chunk.similarity_score, reverse=True)
        return ranked[:top_k]
