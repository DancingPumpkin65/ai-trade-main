from __future__ import annotations

from trading_agents.core.models import NewsChunk
from trading_agents.core.rag.store import BaseVectorStore


BLOCKED_DOMAINS = {"finance.yahoo.com", "nasdaq.com", "seekingalpha.com", "etfdailynews.com", "marketscreener.com"}


class Indexer:
    def __init__(self, store: BaseVectorStore):
        self.store = store

    def upsert_news(self, chunks: list[NewsChunk]) -> int:
        filtered = [chunk for chunk in chunks if not any(domain in (chunk.url or "") for domain in BLOCKED_DOMAINS)]
        self.store.upsert("news", filtered)
        return len(filtered)

    def upsert_macro_documents(self, chunks: list[NewsChunk]) -> int:
        self.store.upsert("macro_documents", chunks)
        return len(chunks)
