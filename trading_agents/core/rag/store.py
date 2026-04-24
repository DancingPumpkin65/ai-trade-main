from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json

from trading_agents.core.models import NewsChunk


class BaseVectorStore:
    backend_name = "base"

    def upsert(self, collection: str, chunks: list[NewsChunk]) -> None:
        raise NotImplementedError

    def query(self, collection: str, query: str, top_k: int = 5, filters: dict | None = None) -> list[NewsChunk]:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    backend_name = "in_memory"

    def __init__(self):
        self.collections: dict[str, dict[str, NewsChunk]] = defaultdict(dict)

    def upsert(self, collection: str, chunks: list[NewsChunk]) -> None:
        for chunk in chunks:
            self.collections[collection][chunk.chunk_id] = chunk

    def query(self, collection: str, query: str, top_k: int = 5, filters: dict | None = None) -> list[NewsChunk]:
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


class ChromaVectorStore(BaseVectorStore):
    backend_name = "chroma"

    def __init__(self, persist_dir: Path):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collections = {
            name: self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
            )
            for name in ("news", "calendar_events", "macro_documents")
        }

    def upsert(self, collection: str, chunks: list[NewsChunk]) -> None:
        if not chunks:
            return
        target = self.collections[collection]
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._chunk_metadata(chunk) for chunk in chunks]
        target.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, collection: str, query: str, top_k: int = 5, filters: dict | None = None) -> list[NewsChunk]:
        target = self.collections[collection]
        result = target.query(
            query_texts=[query],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[NewsChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance or 0.0))
            published_at = self._parse_datetime((metadata or {}).get("published_at"))
            chunks.append(
                NewsChunk(
                    chunk_id=chunk_id,
                    text=document,
                    source=(metadata or {}).get("source", "unknown"),
                    published_at=published_at,
                    similarity_score=similarity,
                    url=(metadata or {}).get("url"),
                    low_confidence=similarity < 0.55,
                    metadata=self._restore_metadata(metadata or {}),
                )
            )
        return chunks

    def _chunk_metadata(self, chunk: NewsChunk) -> dict:
        metadata = dict(chunk.metadata)
        sanitized: dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = [str(item) for item in value]
            else:
                sanitized[key] = json.dumps(value, ensure_ascii=True)
        sanitized["source"] = chunk.source
        sanitized["published_at"] = chunk.published_at.isoformat() if chunk.published_at else ""
        sanitized["url"] = chunk.url or ""
        return sanitized

    def _restore_metadata(self, metadata: dict) -> dict:
        restored = dict(metadata)
        restored.pop("source", None)
        restored.pop("published_at", None)
        restored.pop("url", None)
        return restored

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


def build_vector_store(*, persist_dir: Path, env: str, prefer_chroma: bool = True) -> BaseVectorStore:
    if env == "test" or not prefer_chroma:
        return InMemoryVectorStore()
    try:
        return ChromaVectorStore(persist_dir)
    except Exception:
        return InMemoryVectorStore()
