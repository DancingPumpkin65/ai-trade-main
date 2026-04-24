from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import md5
from pathlib import Path
import re

import httpx
import pdfplumber

from trading_agents.core.models import NewsChunk


@dataclass(slots=True)
class BourseRunSummary:
    indexed_chunks: int
    errors: list[str]


class BourseDataFetcher:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def run_daily(self) -> dict:
        return {"indexed_chunks": 0, "errors": []}

    async def fetch_resume_pdf(self, target_date: date) -> Path | None:
        url = f"https://media.casablanca-bourse.com/sites/default/files/es-auto-upload/fr/resume_seance_{target_date:%Y%m%d}.pdf"
        destination = self.cache_dir / f"resume_seance_{target_date:%Y%m%d}.pdf"
        if destination.exists():
            return destination
        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                destination.write_bytes(response.content)
                return destination
        except Exception:
            return None
        return None

    def extract_chunks(self, pdf_path: Path, doc_type: str = "corporate_notices") -> list[NewsChunk]:
        text_parts: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text_parts.append(text)
        text = "\n".join(text_parts).strip()
        if len(text) < 50:
            return []
        notices = self._extract_notices(text)
        chunks: list[NewsChunk] = []
        for index, notice in enumerate(notices):
            chunks.append(
                NewsChunk(
                    chunk_id=md5(f"{pdf_path.name}|{doc_type}|{index}".encode("utf-8")).hexdigest(),
                    text=notice,
                    source="Casablanca Bourse PDF",
                    published_at=datetime.now(timezone.utc),
                    similarity_score=1.0,
                    metadata={"doc_type": doc_type},
                )
            )
        return chunks

    def _extract_notices(self, text: str) -> list[str]:
        pattern = re.compile(r"(\d{2}/\d{2}/\d{4}.*)")
        notices = [line.strip() for line in text.splitlines() if pattern.match(line.strip())]
        return notices[:20] or [text[:1000]]

    def last_completed_trading_days(self, count: int = 10) -> list[date]:
        days: list[date] = []
        current = date.today() - timedelta(days=1)
        while len(days) < count:
            if current.weekday() < 5:
                days.append(current)
            current -= timedelta(days=1)
        return days
