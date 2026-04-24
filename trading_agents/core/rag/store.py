from __future__ import annotations

from collections import defaultdict

from trading_agents.core.models import NewsChunk


class InMemoryVectorStore:
    def __init__(self):
        self.collections: dict[str, dict[str, NewsChunk]] = defaultdict(dict)

    def upsert(self, collection: str, chunks: list[NewsChunk]) -> None:
        for chunk in chunks:
            self.collections[collection][chunk.chunk_id] = chunk

    def query(self, collection: str, query: str, top_k: int = 5) -> list[NewsChunk]:
        tokens = {token.lower() for token in query.split() if token}
        ranked: list[tuple[float, NewsChunk]] = []
        for chunk in self.collections.get(collection, {}).values():
            haystack = f"{chunk.source} {chunk.text}".lower()
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap == 0:
                continue
            score = min(1.0, overlap / max(len(tokens), 1) + 0.3)
            ranked.append((score, chunk.model_copy(update={"similarity_score": score, "low_confidence": score < 0.55})))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
