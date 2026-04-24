from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sin

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trading_agents.core.models import StockInfo


SAMPLE_STOCKS = [
    {"symbol": "ATW", "name": "Attijariwafa Bank", "sector": "Banks", "market_cap": 98_000_000_000, "price": 520.0},
    {"symbol": "IAM", "name": "Maroc Telecom", "sector": "Telecom", "market_cap": 84_000_000_000, "price": 112.0},
    {"symbol": "BCP", "name": "Banque Centrale Populaire", "sector": "Banks", "market_cap": 54_000_000_000, "price": 275.0},
    {"symbol": "MNG", "name": "Managem", "sector": "Mining", "market_cap": 41_000_000_000, "price": 2_110.0},
    {"symbol": "CIH", "name": "CIH Bank", "sector": "Banks", "market_cap": 18_000_000_000, "price": 381.0},
]


class DrahmiClient:
    def __init__(self, base_url: str, api_key: str | None, daily_limit: int = 500):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.daily_limit = daily_limit
        self.request_counter: defaultdict[str, int] = defaultdict(int)

    def _check_limit(self) -> None:
        key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.request_counter[key] += 1
        if self.request_counter[key] > self.daily_limit:
            raise RuntimeError("Drahmi daily request limit reached.")

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        self._check_limit()
        if not self.api_key:
            raise RuntimeError("DRAHMI_API_KEY is not configured.")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}{path}", params=params, headers=headers)
        if response.status_code in {401, 403, 404}:
            raise RuntimeError(f"Drahmi request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()

    async def list_stocks(self) -> list[StockInfo]:
        if not self.api_key:
            return [self._sample_to_stock(item) for item in SAMPLE_STOCKS]
        page = 1
        stocks: list[StockInfo] = []
        while True:
            payload = await self._get("/stocks", {"page": page})
            data = payload if isinstance(payload, list) else payload.get("data", [])
            if not data:
                break
            for item in data:
                stocks.append(self._payload_to_stock(item))
            page += 1
        return stocks

    async def get_stock(self, symbol: str) -> StockInfo:
        symbol = symbol.upper()
        if not self.api_key:
            sample = next((item for item in SAMPLE_STOCKS if item["symbol"] == symbol), None)
            if sample is None:
                sample = {"symbol": symbol, "name": symbol, "sector": "Unknown", "market_cap": 1_000_000_000, "price": 100.0}
            return self._sample_to_stock(sample)
        stock_payload = await self._get(f"/stocks/{symbol}")
        history_payload = await self._get(f"/stocks/{symbol}/history")
        stock = self._payload_to_stock(stock_payload)
        stock.ohlcv = self._normalize_history(history_payload)
        return stock

    def _payload_to_stock(self, payload: dict) -> StockInfo:
        symbol = payload.get("ticker") or payload.get("symbol") or "UNKNOWN"
        stock = StockInfo(
            symbol=symbol,
            name=payload.get("name", symbol),
            sector=payload.get("sector", "Unknown"),
            market_cap=float(payload.get("market_cap") or 0.0),
            last_price=float(payload.get("last_price") or payload.get("price") or 0.0),
            last_volume=float(payload.get("last_volume") or payload.get("volume") or 0.0),
            high_52w=float(payload.get("high_52w") or payload.get("fifty_two_week_high") or 0.0),
            low_52w=float(payload.get("low_52w") or payload.get("fifty_two_week_low") or 0.0),
            ohlcv=self._normalize_history(payload.get("history", [])),
        )
        return stock

    def _sample_to_stock(self, item: dict) -> StockInfo:
        return StockInfo(
            symbol=item["symbol"],
            name=item["name"],
            sector=item["sector"],
            market_cap=float(item["market_cap"]),
            last_price=float(item["price"]),
            last_volume=2_500_000.0,
            high_52w=float(item["price"]) * 1.15,
            low_52w=float(item["price"]) * 0.75,
            ohlcv=self._sample_history(float(item["price"])),
        )

    def _sample_history(self, base_price: float) -> list[dict]:
        bars = []
        today = datetime.now(timezone.utc).date()
        for offset in range(30, 0, -1):
            day = today - timedelta(days=offset)
            close = base_price * (1 + sin(offset / 4) * 0.02)
            open_price = close * 0.997
            high = close * 1.01
            low = close * 0.99
            bars.append(
                {
                    "date": day.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": 10000 + offset * 150,
                }
            )
        return bars

    def _normalize_history(self, payload: list | dict) -> list[dict]:
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "date": row.get("date") or row.get("day"),
                    "open": float(row.get("open") or row.get("o") or 0.0),
                    "high": float(row.get("high") or row.get("h") or 0.0),
                    "low": float(row.get("low") or row.get("l") or 0.0),
                    "close": float(row.get("close") or row.get("c") or 0.0),
                    "volume": float(row.get("volume") or row.get("v") or 0.0),
                }
            )
        return normalized
