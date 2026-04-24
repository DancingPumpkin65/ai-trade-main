import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

from trading_agents.core.broker.alpaca import AlpacaPreviewService
from trading_agents.core.data.bourse_fetcher import BourseDataFetcher
from trading_agents.core.data.drahmi import DrahmiClient
from trading_agents.core.data.news_global import MarketAuxClient
from trading_agents.core.data.news_morocco import MoroccoNewsClient
from trading_agents.core.models import NewsChunk
from trading_agents.core.storage import Storage
from trading_agents.graph.build import TradingGraphService


SAMPLE_DAILY_TEXT = """
MASI: 17 234,12
Variation journaliere: +0,80%
Performance depuis janvier: +4,20%
Volume global: 123 456 789
MASI BANQUES 12 345,67 -1,92% -7,60%
ATTIJARIWAFA BANK 510,00 520,00 +1,96% 12 000 000 15 000
IAM 100,00 98,00 -2,00% 8 000 000 22 000
12/04/2026 Publication de l'avis de mise en paiement du dividende
13/04/2026 Avis de convocation a l'assemblee generale
"""


def test_bourse_fetcher_builds_expected_daily_chunks(tmp_path: Path):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    chunks = fetcher.extract_chunks_from_text(
        text=SAMPLE_DAILY_TEXT,
        source_key="resume_seance_20260412.pdf",
        target_date=date(2026, 4, 12),
        period_type="daily",
    )
    doc_types = {chunk.metadata.get("doc_type") for chunk in chunks}
    assert "market_overview" in doc_types
    assert "sector_indices" in doc_types
    assert "top_movers" in doc_types
    assert "stock_performance" in doc_types
    assert "corporate_notices" in doc_types

    notices = [chunk for chunk in chunks if chunk.metadata.get("doc_type") == "corporate_notices"][0]
    assert "mise en paiement du dividende" in notices.text


def test_bourse_fetcher_generates_period_targets(tmp_path: Path):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    assert len(fetcher._daily_targets(10)) == 10
    assert len(fetcher._weekly_targets(2)) == 2
    assert len(fetcher._monthly_targets(2)) == 2
    assert len(fetcher._quarterly_targets(2)) == 2


def test_graph_service_auto_indexes_bourse_chunks(tmp_path: Path):
    storage = Storage(tmp_path / "trading.db")
    graph_service = TradingGraphService(
        storage=storage,
        drahmi_client=DrahmiClient("https://api.drahmi.app/api", api_key=None, daily_limit=500),
        morocco_news_client=MoroccoNewsClient(),
        marketaux_client=MarketAuxClient("https://api.marketaux.com/v1", api_key=None),
        alpaca_preview_service=AlpacaPreviewService(enabled=False),
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        chroma_persist_dir=tmp_path / "chroma",
        bourse_cache_dir=tmp_path / "bourse_cache",
        env="dev",
    )

    async def fake_run_daily():
        return {
            "indexed_chunks": 1,
            "processed_files": 1,
            "errors": [],
            "chunks": [
                NewsChunk(
                    chunk_id="macro-1",
                    text="ATW corporate notice dividend payment Morocco",
                    source="Casablanca Bourse PDF",
                    published_at=datetime.now(timezone.utc),
                    similarity_score=1.0,
                    metadata={"doc_type": "corporate_notices"},
                )
            ],
        }

    graph_service.bourse_fetcher.run_daily = fake_run_daily
    asyncio.run(graph_service.ingest_bourse_documents())
    results = graph_service.retriever.search_news("ATW dividend Morocco", top_k=5)
    assert any(chunk.chunk_id == "macro-1" for chunk in results)
