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

SAMPLE_ISSUER_HTML = """
<div class="p-[25px] flex flex-col h-full">
  <div class="flex md:items-center md:flex-row mb-4 flex-col-reverse">
    <p class="text-lg leading-[25px] font-bold text-primary-600 hover:text-primary-300">ATTIJARIWAFA BANK</p>
    <p class="text-gray-300 text-xs leading-[18px] whitespace-nowrap md:px-[18px] md:ml-[18px] md:border-l md:border-l-gray-300 mb-2 md:mb-0">15/04/2026</p>
  </div>
  <a class="appearance-none text-sm leading-[21px] block" href="https://media.casablanca-bourse.com/sites/default/files/2026-04/atw_dividende.pdf">
    <h3 class="text-xl leading-[30px] font-semibold text-primary-900 mb-4 hover:text-primary-500 text-left">Attijariwafa Bank - Avis de mise en paiement du dividende 2025</h3>
  </a>
</div>
<div class="p-[25px] flex flex-col h-full">
  <div class="flex md:items-center md:flex-row mb-4 flex-col-reverse">
    <p class="text-lg leading-[25px] font-bold text-primary-600 hover:text-primary-300">ITISSALAT AL-MAGHRIB</p>
    <p class="text-gray-300 text-xs leading-[18px] whitespace-nowrap md:px-[18px] md:ml-[18px] md:border-l md:border-l-gray-300 mb-2 md:mb-0">14/04/2026</p>
  </div>
  <a class="appearance-none text-sm leading-[21px] block" href="https://media.casablanca-bourse.com/sites/default/files/2026-04/iam_buyback.pdf">
    <h3 class="text-xl leading-[30px] font-semibold text-primary-900 mb-4 hover:text-primary-500 text-left">Maroc Telecom - notice d'information relative au programme de rachat d'actions</h3>
  </a>
</div>
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


def test_bourse_fetcher_parses_and_matches_issuer_publications(tmp_path: Path):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    entries = fetcher.parse_issuer_publications_page(SAMPLE_ISSUER_HTML)
    assert len(entries) == 2

    matched = fetcher.match_issuer_publications(
        entries,
        [
            DrahmiClient("https://api.drahmi.app/api", api_key=None, daily_limit=500)._sample_to_stock(
                {"symbol": "ATW", "name": "Attijariwafa Bank", "sector": "Banks", "market_cap": 1, "price": 1}
            ),
            DrahmiClient("https://api.drahmi.app/api", api_key=None, daily_limit=500)._sample_to_stock(
                {"symbol": "IAM", "name": "Maroc Telecom", "sector": "Telecom", "market_cap": 1, "price": 1}
            ),
        ],
    )

    tickers = {entry.ticker for entry in matched}
    assert tickers == {"ATW", "IAM"}
    assert matched[0].published_date >= matched[1].published_date


def test_bourse_fetcher_chunks_issuer_publication_pages(tmp_path: Path):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    chunks = fetcher.extract_issuer_publication_chunks_from_pages(
        pages=[
            "Page 1 resultats financiers 2025",
            "Page 2 dividende propose et calendrier",
            "Page 3 details complementaires",
            "Page 4 structure financiere",
            "Page 5 perspectives 2026",
            "Page 6 annexes",
        ],
        source_key="atw_dividende.pdf",
        ticker="ATW",
        issuer_name="Attijariwafa Bank",
        title="Attijariwafa Bank - Avis de mise en paiement du dividende 2025",
        published_date=date(2026, 4, 15),
        publication_url="https://media.casablanca-bourse.com/sites/default/files/2026-04/atw_dividende.pdf",
    )

    assert len(chunks) == 3
    assert chunks[0].metadata["doc_type"] == "issuer_publication"
    assert chunks[0].metadata["chunk_role"] == "summary"
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2
    assert "Page 1 resultats financiers 2025" in chunks[0].text
    assert chunks[1].metadata["page_start"] == 3
    assert chunks[1].metadata["page_end"] == 5
    assert chunks[2].metadata["page_start"] == 6
    assert chunks[2].metadata["page_end"] == 6


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


def test_graph_service_auto_indexes_issuer_publication_chunks(tmp_path: Path):
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
        env="test",
    )

    async def fake_fetch_issuer_publication_chunks(stocks, limit_pdfs=3):
        assert stocks[0].symbol == "ATW"
        return [
            NewsChunk(
                chunk_id="issuer-1",
                text="Publication emetteur — ATW — dividende 2025",
                source="Casablanca Bourse Issuer Publication",
                published_at=datetime.now(timezone.utc),
                similarity_score=1.0,
                url="https://media.casablanca-bourse.com/sites/default/files/2026-04/atw_dividende.pdf",
                metadata={"doc_type": "issuer_publication", "ticker": "ATW"},
            )
        ]

    graph_service.bourse_fetcher.fetch_issuer_publication_chunks = fake_fetch_issuer_publication_chunks
    asyncio.run(graph_service.ingest_news("ATW"))
    results = graph_service.retriever.search_news("ATW dividende 2025", top_k=5)
    assert any(chunk.chunk_id == "issuer-1" for chunk in results)
