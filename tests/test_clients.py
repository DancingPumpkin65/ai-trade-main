from __future__ import annotations

from datetime import date

import httpx
import pytest

from trading_agents.core.data.bourse_fetcher import BourseDataFetcher, PdfTarget
from trading_agents.core.data.drahmi import DrahmiClient, DrahmiSchemaError
from trading_agents.core.data.news_global import MarketAuxClient
from trading_agents.core.models import StockInfo


class FakeResponse:
    def __init__(self, *, url: str, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.request = httpx.Request("GET", url)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
                text=self.text,
                json=self._payload if isinstance(self._payload, dict) else None,
            )
            raise httpx.HTTPStatusError("Request failed", request=self.request, response=response)


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse] | None = None, exception: Exception | None = None, **kwargs):
        self.responses = list(responses or [])
        self.exception = exception

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, params=None, headers=None, follow_redirects=False):
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            raise AssertionError(f"No fake response queued for {url}")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_drahmi_get_raises_runtime_error_on_401(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)
    responses = [FakeResponse(url="https://api.drahmi.app/api/stocks", status_code=401, text="unauthorized")]
    monkeypatch.setattr("trading_agents.core.data.drahmi.httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient(responses=responses))

    with pytest.raises(RuntimeError, match="status 401"):
        await client._get("/stocks")


@pytest.mark.asyncio
async def test_drahmi_get_raises_runtime_error_on_404(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)
    responses = [FakeResponse(url="https://api.drahmi.app/api/stocks/ATW", status_code=404, text="not found")]
    monkeypatch.setattr("trading_agents.core.data.drahmi.httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient(responses=responses))

    with pytest.raises(RuntimeError, match="status 404"):
        await client._get("/stocks/ATW")


@pytest.mark.asyncio
async def test_drahmi_list_stocks_raises_schema_error_on_missing_data_array(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)

    async def fake_get(path: str, params: dict | None = None):
        return {"items": [{"symbol": "ATW", "price": 520.0}]}

    monkeypatch.setattr(client, "_get", fake_get)

    with pytest.raises(DrahmiSchemaError, match="must contain a 'data' array"):
        await client.list_stocks()


@pytest.mark.asyncio
async def test_drahmi_get_stock_raises_schema_error_on_missing_symbol(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)

    async def fake_get(path: str, params: dict | None = None):
        if path.endswith("/history"):
            return {"data": [{"date": "2026-04-24", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1234}]}
        return {"name": "Attijariwafa Bank", "price": 520.0}

    monkeypatch.setattr(client, "_get", fake_get)

    with pytest.raises(DrahmiSchemaError, match="missing required field 'symbol'"):
        await client.get_stock("ATW")


@pytest.mark.asyncio
async def test_drahmi_get_stock_raises_schema_error_on_malformed_history_row(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)

    async def fake_get(path: str, params: dict | None = None):
        if path.endswith("/history"):
            return {"data": [{"date": "2026-04-24", "open": 100, "high": 101, "low": 99, "volume": 1234}]}
        return {"symbol": "ATW", "name": "Attijariwafa Bank", "price": 520.0}

    monkeypatch.setattr(client, "_get", fake_get)

    with pytest.raises(DrahmiSchemaError, match="missing required field 'close'"):
        await client.get_stock("ATW")


@pytest.mark.asyncio
async def test_drahmi_get_stock_accepts_valid_string_numeric_payloads(monkeypatch):
    client = DrahmiClient("https://api.drahmi.app/api", api_key="secret", daily_limit=10)

    async def fake_get(path: str, params: dict | None = None):
        if path.endswith("/history"):
            return {
                "data": [
                    {"day": "2026-04-24", "o": "100.1", "h": "101.2", "l": "99.8", "c": "100.7", "v": "1500"},
                    {"day": "2026-04-25", "o": "100.7", "h": "102.0", "l": "100.0", "c": "101.5", "v": "1700"},
                ]
            }
        return {
            "ticker": "ATW",
            "name": "Attijariwafa Bank",
            "sector": "Banks",
            "market_cap": "98000000000",
            "last_price": "520.0",
            "last_volume": "2500000",
            "high_52w": "560.0",
            "low_52w": "430.0",
            "market_mode": "continuous",
        }

    monkeypatch.setattr(client, "_get", fake_get)

    stock = await client.get_stock("ATW")

    assert stock.symbol == "ATW"
    assert stock.last_price == 520.0
    assert stock.ohlcv[0]["open"] == 100.1
    assert len(stock.ohlcv) == 2


@pytest.mark.asyncio
async def test_marketaux_returns_empty_list_on_401_and_429(monkeypatch):
    queued = [
        FakeResponse(url="https://api.marketaux.com/v1/news/all", status_code=401, text="unauthorized"),
        FakeResponse(url="https://api.marketaux.com/v1/news/all", status_code=429, text="rate limited"),
    ]

    def fake_async_client(*args, **kwargs):
        return FakeAsyncClient(responses=[queued.pop(0)])

    monkeypatch.setattr("trading_agents.core.data.news_global.httpx.AsyncClient", fake_async_client)
    client = MarketAuxClient("https://api.marketaux.com/v1", api_key="secret")

    assert await client.fetch_for_symbol("ATW") == []
    assert await client.fetch_for_symbol("ATW") == []


@pytest.mark.asyncio
async def test_marketaux_raises_on_server_error(monkeypatch):
    responses = [FakeResponse(url="https://api.marketaux.com/v1/news/all", status_code=500, text="server error")]
    monkeypatch.setattr("trading_agents.core.data.news_global.httpx.AsyncClient", lambda *args, **kwargs: FakeAsyncClient(responses=responses))
    client = MarketAuxClient("https://api.marketaux.com/v1", api_key="secret")

    with pytest.raises(httpx.HTTPStatusError):
        await client.fetch_for_symbol("ATW")


@pytest.mark.asyncio
async def test_bourse_run_daily_collects_missing_pdf_errors(tmp_path):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    target = PdfTarget(
        period_type="daily",
        target_date=date(2026, 4, 24),
        url="https://media.casablanca-bourse.com/sites/default/files/es-auto-upload/fr/resume_seance_20260424.pdf",
        filename="resume_seance_20260424.pdf",
    )

    fetcher._daily_targets = lambda count: [target]
    fetcher._weekly_targets = lambda count: []
    fetcher._monthly_targets = lambda count: []
    fetcher._quarterly_targets = lambda count: []

    async def fake_download_if_needed(client, current_target):
        return None

    fetcher._download_if_needed = fake_download_if_needed
    summary = await fetcher.run_daily()
    assert summary["processed_files"] == 0
    assert summary["indexed_chunks"] == 0
    assert summary["errors"]
    assert "Missing PDF for daily 2026-04-24" in summary["errors"][0]


@pytest.mark.asyncio
async def test_bourse_issuer_publications_return_empty_on_page_failure(tmp_path, monkeypatch):
    fetcher = BourseDataFetcher(tmp_path / "bourse")
    fake_stock = StockInfo(
        symbol="ATW",
        name="Attijariwafa Bank",
        sector="Banks",
        market_cap=1.0,
        last_price=100.0,
        last_volume=100000.0,
        high_52w=120.0,
        low_52w=80.0,
        ohlcv=[],
    )
    page_error = httpx.ConnectError("network down")
    monkeypatch.setattr(
        "trading_agents.core.data.bourse_fetcher.httpx.AsyncClient",
        lambda *args, **kwargs: FakeAsyncClient(exception=page_error),
    )

    chunks = await fetcher.fetch_issuer_publication_chunks([fake_stock], limit_pdfs=3)
    assert chunks == []
