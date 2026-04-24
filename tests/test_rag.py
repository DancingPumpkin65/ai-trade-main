from datetime import datetime, timezone
from pathlib import Path

from trading_agents.core.models import NewsChunk
from trading_agents.core.rag.indexer import Indexer
from trading_agents.core.rag.retriever import NewsRetriever
from trading_agents.core.rag.store import InMemoryVectorStore, build_vector_store


def test_vector_store_falls_back_to_in_memory_in_test_env(tmp_path: Path):
    store = build_vector_store(persist_dir=tmp_path / "chroma", env="test", prefer_chroma=True)
    assert isinstance(store, InMemoryVectorStore)
    assert store.backend_name == "in_memory"


def test_vector_store_falls_back_when_sentence_transformers_missing(tmp_path: Path):
    store = build_vector_store(persist_dir=tmp_path / "chroma", env="dev", prefer_chroma=True)
    assert store.backend_name in {"in_memory", "chroma"}


def test_retriever_merges_news_and_macro_documents(tmp_path: Path):
    store = InMemoryVectorStore()
    indexer = Indexer(store)
    retriever = NewsRetriever(store)
    now = datetime.now(timezone.utc)

    indexer.upsert_news(
        [
            NewsChunk(
                chunk_id="news-1",
                text="ATW dividend growth Morocco",
                source="Sample News",
                published_at=now,
                url="https://example.com/atw-news",
            )
        ]
    )
    indexer.upsert_macro_documents(
        [
            NewsChunk(
                chunk_id="doc-1",
                text="ATW corporate notice dividend payment Morocco",
                source="Sample Doc",
                published_at=now,
                url="https://example.com/atw-doc",
            )
        ]
    )

    results = retriever.search_news("ATW dividend Morocco", top_k=5, filters=None)
    ids = {item.chunk_id for item in results}
    assert "news-1" in ids
    assert "doc-1" in ids


def test_retriever_logs_search_metadata_to_tracer(tmp_path: Path):
    class FakeTracer:
        def __init__(self):
            self.calls = []

        def log_search(self, **kwargs):
            self.calls.append(kwargs)

    store = InMemoryVectorStore()
    indexer = Indexer(store)
    tracer = FakeTracer()
    retriever = NewsRetriever(store, tracer=tracer)
    now = datetime.now(timezone.utc)

    indexer.upsert_news(
        [
            NewsChunk(
                chunk_id="news-2",
                text="ATW earnings Morocco",
                source="Sample News",
                published_at=now,
                url="https://example.com/atw-earnings",
                metadata={"doc_type": "news"},
            )
        ]
    )

    results = retriever.search_news(
        "ATW earnings",
        top_k=3,
        filters={"ticker": "ATW"},
        metadata={"request_id": "req-123", "symbol": "ATW", "query_source": "sentiment_tool"},
    )

    assert len(results) == 1
    assert len(tracer.calls) == 1
    call = tracer.calls[0]
    assert call["query"] == "ATW earnings"
    assert call["top_k"] == 3
    assert call["filters"] == {"ticker": "ATW"}
    assert call["collections"] == ["news", "macro_documents"]
    assert call["metadata"]["request_id"] == "req-123"
    assert call["metadata"]["symbol"] == "ATW"
    assert call["results"][0].chunk_id == "news-2"
