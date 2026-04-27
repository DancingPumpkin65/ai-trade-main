from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_agents.core.models import NewsChunk, RiskOutput, StockInfo


def _trend_stock(
    symbol: str,
    name: str,
    *,
    start_price: float = 100.0,
    step: float = 1.2,
    last_volume: float = 2_000_000.0,
    bars: int = 30,
) -> StockInfo:
    history = []
    for idx in range(bars):
        close = start_price + step * idx
        history.append(
            {
                "date": f"2026-03-{idx + 1:02d}",
                "open": round(close * 0.995, 2),
                "high": round(close * 1.01, 2),
                "low": round(close * 0.99, 2),
                "close": round(close, 2),
                "volume": last_volume,
            }
        )
    return StockInfo(
        symbol=symbol,
        name=name,
        sector="Banks",
        market_cap=50_000_000_000,
        last_price=history[-1]["close"],
        last_volume=last_volume,
        high_52w=max(bar["close"] for bar in history),
        low_52w=min(bar["close"] for bar in history),
        ohlcv=history,
    )


def _downtrend_stock(
    symbol: str,
    name: str,
    *,
    start_price: float = 160.0,
    step: float = -1.8,
    last_volume: float = 2_100_000.0,
    bars: int = 30,
) -> StockInfo:
    return _trend_stock(symbol, name, start_price=start_price, step=step, last_volume=last_volume, bars=bars)


def _volatile_stock(symbol: str = "ATW") -> StockInfo:
    closes = [100, 135, 82, 140, 78, 145, 74, 150, 70, 155, 68, 160, 66, 162, 64, 165, 62, 168, 60, 170, 58, 172]
    history = []
    for idx, close in enumerate(closes, start=1):
        history.append(
            {
                "date": f"2026-04-{idx:02d}",
                "open": close * 0.96,
                "high": close * 1.04,
                "low": close * 0.94,
                "close": close,
                "volume": 10000 + idx * 250,
            }
        )
    return StockInfo(
        symbol=symbol,
        name="Attijariwafa Bank",
        sector="Banks",
        market_cap=98_000_000_000,
        last_price=closes[-1],
        last_volume=history[-1]["volume"],
        high_52w=max(closes),
        low_52w=min(closes),
        ohlcv=history,
    )


def _positive_chunks(symbol: str, name: str, now: datetime) -> list[NewsChunk]:
    return [
        NewsChunk(
            chunk_id=f"{symbol.lower()}-notice",
            text=f"{name} annonce un dividende exceptionnel et une hausse de resultat.",
            source="Casablanca Bourse PDF",
            published_at=now - timedelta(days=1),
            similarity_score=0.95,
            url=f"https://example.com/{symbol.lower()}-notice.pdf",
            metadata={"doc_type": "corporate_notices", "ticker": symbol},
        ),
        NewsChunk(
            chunk_id=f"{symbol.lower()}-issuer",
            text=f"{name} confirme une croissance solide et un contrat strategique.",
            source="Casablanca Bourse Issuer Publication",
            published_at=now - timedelta(days=2),
            similarity_score=0.9,
            url=f"https://example.com/{symbol.lower()}-issuer.pdf",
            metadata={"doc_type": "issuer_publication", "ticker": symbol},
        ),
    ]


def _negative_chunks(symbol: str, name: str, now: datetime) -> list[NewsChunk]:
    return [
        NewsChunk(
            chunk_id=f"{symbol.lower()}-warning",
            text=f"{name} publie un warning sur la baisse de marge et un recul d'activite.",
            source="Sample Feed",
            published_at=now - timedelta(days=1),
            similarity_score=0.92,
            url=f"https://example.com/{symbol.lower()}-warning",
            metadata={"doc_type": "news", "ticker": symbol},
        ),
        NewsChunk(
            chunk_id=f"{symbol.lower()}-downgrade",
            text=f"{name} subit un downgrade apres un decline operationnel.",
            source="MarketAux",
            published_at=now - timedelta(days=2),
            similarity_score=0.88,
            url=f"https://example.com/{symbol.lower()}-downgrade",
            metadata={"doc_type": "news", "ticker": symbol},
        ),
    ]


def install_fixture(services, fixture_name: str) -> None:
    async def noop_ingest_news(symbol: str | None = None) -> None:
        return None

    async def noop_ingest_bourse_documents() -> None:
        return None

    async def noop_ingest_issuer_publications(symbol: str) -> None:
        return None

    services.graph_service.ingest_news = noop_ingest_news
    services.graph_service.ingest_bourse_documents = noop_ingest_bourse_documents
    services.graph_service.ingest_issuer_publications = noop_ingest_issuer_publications

    now = datetime.now(timezone.utc)

    if fixture_name == "single_symbol_conservative":
        stock = _trend_stock("ATW", "Attijariwafa Bank", step=1.9, last_volume=2_500_000.0)

        async def get_stock(symbol: str):
            return stock

        def search_news(query: str, top_k: int = 5, filters=None, metadata=None):
            return _positive_chunks("ATW", "Attijariwafa Bank", now)

        services.drahmi_client.get_stock = get_stock
        services.graph_service.retriever.search_news = search_news
        return

    if fixture_name == "universe_scan_weekly":
        universe = {
            "ATW": _trend_stock("ATW", "Attijariwafa Bank", step=1.8, last_volume=2_600_000.0),
            "IAM": _trend_stock("IAM", "Maroc Telecom", step=-0.45, last_volume=300_000.0),
            "LHM": _trend_stock("LHM", "LafargeHolcim Maroc", step=-0.2, last_volume=90_000.0),
        }

        async def list_stocks():
            return list(universe.values())

        async def get_stock(symbol: str):
            return universe[symbol.upper()]

        def search_news(query: str, top_k: int = 8, filters=None, metadata=None):
            query_upper = query.upper()
            if "ATW" in query_upper:
                return _positive_chunks("ATW", "Attijariwafa Bank", now)
            if "IAM" in query_upper:
                return [
                    NewsChunk(
                        chunk_id="iam-old",
                        text="Maroc Telecom publie une note generale plus ancienne.",
                        source="Sample Feed",
                        published_at=now - timedelta(days=20),
                        similarity_score=0.6,
                        url="https://example.com/iam-old",
                        metadata={"doc_type": "news", "ticker": "IAM"},
                    )
                ]
            return []

        services.drahmi_client.list_stocks = list_stocks
        services.drahmi_client.get_stock = get_stock
        services.graph_service.retriever.search_news = search_news
        return

    if fixture_name == "forced_buy_bias_refused":
        stock = _downtrend_stock("ATW", "Attijariwafa Bank")

        async def get_stock(symbol: str):
            return stock

        def search_news(query: str, top_k: int = 5, filters=None, metadata=None):
            return _negative_chunks("ATW", "Attijariwafa Bank", now)

        services.drahmi_client.get_stock = get_stock
        services.graph_service.retriever.search_news = search_news
        return

    if fixture_name == "high_risk_review":
        stock = _volatile_stock("ATW")
        services.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")
        services.graph_service.alpaca_preview_service.register_symbol_mapping("ATW", "SPY")

        async def get_stock(symbol: str):
            return stock

        def search_news(query: str, top_k: int = 5, filters=None, metadata=None):
            return _positive_chunks("ATW", "Attijariwafa Bank", now)

        def forced_risk_loop(*, request_id: str, symbol: str, capital: float, request_intent, scratchpad: list[dict]):
            messages = list(scratchpad)
            messages.append({"role": "assistant", "content": "Forced risk output for harness order-approval scenario."})
            return (
                RiskOutput(
                    action="BUY",
                    position_size_pct=0.03,
                    position_value_mad=capital * 0.03,
                    stop_loss_pct=0.08,
                    take_profit_pct=0.12,
                    risk_score=0.72,
                    volatility_estimate=0.65,
                    rationale="Forced BUY risk output for harness order-approval scenario.",
                ),
                messages,
                [],
            )

        services.drahmi_client.get_stock = get_stock
        services.graph_service.retriever.search_news = search_news
        services.graph_service._run_risk_loop = forced_risk_loop
        return

    raise ValueError(f"Unknown harness fixture: {fixture_name}")
