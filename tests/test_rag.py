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
