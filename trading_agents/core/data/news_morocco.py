from __future__ import annotations

from datetime import datetime, timezone
from hashlib import md5
from time import mktime

import feedparser
import httpx

from trading_agents.core.models import NewsChunk


RSS_FEEDS = {
    "L'Economiste": ["https://leseco.ma/feed", "https://leseco.ma/feed/"],
    "Hespress FR": ["https://fr.hespress.com/feed", "https://fr.hespress.com/feed/"],
    "Aujourd'hui Le Maroc": ["https://aujourdhui.ma/feed/", "https://aujourdhui.ma/feed"],
    "Médias24": ["https://medias24.com/feed/", "https://medias24.com/feed"],
}


class MoroccoNewsClient:
    async def fetch(self) -> list[NewsChunk]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            chunks: list[NewsChunk] = []
            for source, urls in RSS_FEEDS.items():
                for url in urls:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        parsed = feedparser.parse(response.text)
                        chunks.extend(self._entries_to_chunks(source, parsed.entries))
                        break
                    except Exception:
                        continue
        if chunks:
            return chunks
        return [
            NewsChunk(
                chunk_id="sample-atw-dividend",
                text="ATW annonce un dividende stable et une dynamique commerciale solide au Maroc.",
                source="Sample Feed",
                published_at=datetime.now(timezone.utc),
                similarity_score=0.8,
                url="https://example.com/atw",
                metadata={"language": "fr", "tags": ["Morocco", "ATW"]},
            ),
            NewsChunk(
                chunk_id="sample-iam-cashflow",
                text="IAM maintient des flux de trésorerie robustes et une politique de distribution régulière.",
                source="Sample Feed",
                published_at=datetime.now(timezone.utc),
                similarity_score=0.75,
                url="https://example.com/iam",
                metadata={"language": "fr", "tags": ["Morocco", "IAM"]},
            ),
        ]

    def _entries_to_chunks(self, source: str, entries: list) -> list[NewsChunk]:
        chunks: list[NewsChunk] = []
        for entry in entries:
            title = getattr(entry, "title", "") or ""
            if not title:
                continue
            body = getattr(entry, "summary", "") or title
            published = None
            parsed_time = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if parsed_time:
                published = datetime.fromtimestamp(mktime(parsed_time), timezone.utc)
            chunk_id = md5(f"{source}|{title}|{published}".encode("utf-8")).hexdigest()
            chunks.append(
                NewsChunk(
                    chunk_id=chunk_id,
                    text=body,
                    source=source,
                    published_at=published,
                    url=getattr(entry, "link", None),
                    metadata={"language": "fr", "tags": ["Morocco"]},
                )
            )
        return chunks
