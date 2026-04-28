from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sin

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trading_agents.core.models import MarketMode, StockInfo


SAMPLE_STOCKS = [
    {"symbol": "ATW", "name": "Attijariwafa Bank", "sector": "Banks", "market_cap": 98_000_000_000, "price": 520.0},
    {"symbol": "IAM", "name": "Maroc Telecom", "sector": "Telecom", "market_cap": 84_000_000_000, "price": 112.0},
    {"symbol": "BCP", "name": "Banque Centrale Populaire", "sector": "Banks", "market_cap": 54_000_000_000, "price": 275.0},
    {"symbol": "MNG", "name": "Managem", "sector": "Mining", "market_cap": 41_000_000_000, "price": 2_110.0},
    {"symbol": "CIH", "name": "CIH Bank", "sector": "Banks", "market_cap": 18_000_000_000, "price": 381.0},
]


class DrahmiClientError(RuntimeError):
    pass


class DrahmiAuthError(DrahmiClientError):
    pass


class DrahmiNotFoundError(DrahmiClientError):
    pass


class DrahmiSchemaError(DrahmiClientError):
    pass


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
            raise DrahmiClientError("DRAHMI_API_KEY is not configured.")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}{path}", params=params, headers=headers)
        if response.status_code in {401, 403}:
            raise DrahmiAuthError(f"Drahmi request failed with status {response.status_code}: {response.text}")
        if response.status_code == 404:
            raise DrahmiNotFoundError(f"Drahmi request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise DrahmiSchemaError(f"Drahmi returned invalid JSON for path '{path}'.") from exc

    async def list_stocks(self) -> list[StockInfo]:
        if not self.api_key:
            return [self._sample_to_stock(item) for item in SAMPLE_STOCKS]
        page = 1
        stocks: list[StockInfo] = []
        while True:
            payload = await self._get("/stocks", {"page": page})
            data = self._coerce_stock_rows(payload, context=f"/stocks page {page}")
            if not data:
                break
            for item in data:
                stocks.append(self._payload_to_stock(item, context=f"/stocks page {page}"))
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
        stock = self._payload_to_stock(stock_payload, context=f"/stocks/{symbol}")
        stock.ohlcv = self._normalize_history(
            history_payload,
            context=f"/stocks/{symbol}/history",
            allow_empty=False,
        )
        return stock

    def _payload_to_stock(self, payload: dict, *, context: str) -> StockInfo:
        if not isinstance(payload, dict):
            raise DrahmiSchemaError(f"{context} must be an object.")
        symbol = self._require_string(payload, ("ticker", "symbol"), context=context, label="symbol")
        last_price = self._require_float(payload, ("last_price", "price"), context=context, label="last_price")
        market_mode = self._extract_market_mode(payload)
        stock = StockInfo(
            symbol=symbol,
            name=payload.get("name", symbol),
            sector=payload.get("sector", "Unknown"),
            market_cap=self._optional_float(payload, ("market_cap",), default=0.0, context=context),
            last_price=last_price,
            last_volume=self._optional_float(payload, ("last_volume", "volume"), default=0.0, context=context),
            high_52w=self._optional_float(payload, ("high_52w", "fifty_two_week_high"), default=0.0, context=context),
            low_52w=self._optional_float(payload, ("low_52w", "fifty_two_week_low"), default=0.0, context=context),
            market_mode=market_mode,
            market_metadata={
                "market_mode": payload.get("market_mode"),
                "trading_mode": payload.get("trading_mode"),
                "quote_mode": payload.get("quote_mode"),
                "instrument_type": payload.get("instrument_type"),
                "asset_type": payload.get("asset_type"),
                "compartment": payload.get("compartment"),
            },
            ohlcv=self._normalize_history(
                payload.get("history", []),
                context=f"{context} inline history",
                allow_empty=True,
            ),
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
            market_mode=MarketMode.CONTINUOUS,
            market_metadata={"market_mode": "continuous"},
            ohlcv=self._sample_history(float(item["price"])),
        )

    def _extract_market_mode(self, payload: dict) -> MarketMode:
        haystack = " ".join(
            str(payload.get(key, ""))
            for key in (
                "market_mode",
                "trading_mode",
                "quote_mode",
                "instrument_type",
                "asset_type",
                "compartment",
                "sector",
                "name",
            )
        ).lower()
        if any(token in haystack for token in ("bond", "obligation", "fixed income", "treasury", "debt")):
            return MarketMode.BOND
        if any(token in haystack for token in ("fixing", "fixe")):
            return MarketMode.FIXING
        if any(token in haystack for token in ("continuous", "continu")):
            return MarketMode.CONTINUOUS
        return MarketMode.UNKNOWN

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

    def _coerce_stock_rows(self, payload: list | dict, *, context: str) -> list[dict]:
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            if data is None:
                raise DrahmiSchemaError(f"{context} must contain a 'data' array.")
            rows = data
        else:
            raise DrahmiSchemaError(f"{context} must be an array or object with a 'data' array.")
        if not isinstance(rows, list):
            raise DrahmiSchemaError(f"{context} data must be an array.")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DrahmiSchemaError(f"{context} row {index} must be an object.")
        return rows

    def _normalize_history(self, payload: list | dict, *, context: str, allow_empty: bool) -> list[dict]:
        rows = self._coerce_history_rows(payload, context=context)
        if not rows and not allow_empty:
            raise DrahmiSchemaError(f"{context} returned no history rows.")
        normalized = []
        for index, row in enumerate(rows):
            normalized.append(
                {
                    "date": self._require_string(row, ("date", "day"), context=f"{context} row {index}", label="date"),
                    "open": self._require_float(row, ("open", "o"), context=f"{context} row {index}", label="open"),
                    "high": self._require_float(row, ("high", "h"), context=f"{context} row {index}", label="high"),
                    "low": self._require_float(row, ("low", "l"), context=f"{context} row {index}", label="low"),
                    "close": self._require_float(row, ("close", "c"), context=f"{context} row {index}", label="close"),
                    "volume": self._optional_float(row, ("volume", "v"), default=0.0, context=f"{context} row {index}"),
                }
            )
        return normalized

    def _coerce_history_rows(self, payload: list | dict, *, context: str) -> list[dict]:
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            if data is None:
                raise DrahmiSchemaError(f"{context} must contain a 'data' array.")
            rows = data
        else:
            raise DrahmiSchemaError(f"{context} must be an array or object with a 'data' array.")
        if not isinstance(rows, list):
            raise DrahmiSchemaError(f"{context} data must be an array.")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DrahmiSchemaError(f"{context} row {index} must be an object.")
        return rows

    def _require_string(self, payload: dict, keys: tuple[str, ...], *, context: str, label: str) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise DrahmiSchemaError(f"{context} is missing required field '{label}'.")

    def _require_float(self, payload: dict, keys: tuple[str, ...], *, context: str, label: str) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise DrahmiSchemaError(f"{context} field '{key}' must be numeric.") from exc
        raise DrahmiSchemaError(f"{context} is missing required field '{label}'.")

    def _optional_float(self, payload: dict, keys: tuple[str, ...], *, default: float, context: str) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise DrahmiSchemaError(f"{context} field '{key}' must be numeric.") from exc
        return default
