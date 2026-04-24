from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import md5

import httpx

from trading_agents.core.models import NewsChunk


class MarketAuxClient:
    def __init__(self, base_url: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def fetch_for_symbol(self, symbol: str) -> list[NewsChunk]:
        if not self.api_key:
            return []
        params = {
            "api_token": self.api_key,
            "search": f"{symbol} Morocco",
            "language": "en",
            "published_after": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "published_before": datetime.now(timezone.utc).isoformat(),
            "limit": 3,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.base_url}/news/all", params=params)
        if response.status_code in {401, 429}:
            return []
        response.raise_for_status()
        payload = response.json()
        chunks: list[NewsChunk] = []
        for item in payload.get("data", []):
            title = item.get("title", "")
            body = item.get("description") or title
            published = item.get("published_at")
            chunk_id = md5(f"MarketAux|{title}|{published}".encode("utf-8")).hexdigest()
            chunks.append(
                NewsChunk(
                    chunk_id=chunk_id,
                    text=body,
                    source="MarketAux",
                    published_at=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None,
                    url=item.get("url"),
                )
            )
        return chunks
