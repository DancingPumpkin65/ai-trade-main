from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import md5
from pathlib import Path
import calendar
import re

import httpx
import pdfplumber

from trading_agents.core.models import NewsChunk


NOTICE_DATE_PATTERN = re.compile(r"^(?P<date>\d{2}/\d{2}/\d{4})\s*(?P<text>.*)$")
STOCK_ROW_PATTERN = re.compile(
    r"^(?P<name>[A-ZÀ-ÖØ-Ý0-9' .&/\-]{3,})\s+(?P<ref>[\d\s.,]+)\s+(?P<close>[\d\s.,]+)\s+(?P<pct>[+\-]?[\d\s.,]+)%\s+(?P<vol>[\d\s.,]+)\s+(?P<qty>[\d\s.,]+)$"
)
SECTOR_ROW_PATTERN = re.compile(
    r"^(?P<sector>MASI\s+[A-ZÀ-ÖØ-Ý /-]+)\s+(?P<value>[\d\s.,]+)\s+(?P<daily>[+\-]?[\d\s.,]+)%\s+(?P<ytd>[+\-]?[\d\s.,]+)%$"
)
MASI_PATTERN = re.compile(r"MASI\s*[:\-]?\s*(?P<value>[\d\s.,]+)")
PERCENT_PATTERN = re.compile(r"(?P<value>[+\-]?[\d\s.,]+)%")
VOLUME_PATTERN = re.compile(r"volume(?:\s+global)?\s*[:\-]?\s*(?P<value>[\d\s.,]+)", re.IGNORECASE)


@dataclass(slots=True)
class PdfTarget:
    period_type: str
    target_date: date
    url: str
    filename: str


@dataclass(slots=True)
class BourseRunSummary:
    indexed_chunks: int
    processed_files: int
    errors: list[str]
    chunks: list[NewsChunk]

    def as_dict(self) -> dict:
        return {
            "indexed_chunks": self.indexed_chunks,
            "processed_files": self.processed_files,
            "errors": self.errors,
            "chunks": self.chunks,
        }


class BourseDataFetcher:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def run_daily(self) -> dict:
        targets = (
            self._daily_targets(10)
            + self._weekly_targets(2)
            + self._monthly_targets(2)
            + self._quarterly_targets(2)
        )
        chunks: list[NewsChunk] = []
        errors: list[str] = []
        processed_files = 0
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            for target in targets:
                pdf_path = await self._download_if_needed(client, target)
                if pdf_path is None:
                    errors.append(f"Missing PDF for {target.period_type} {target.target_date.isoformat()}")
                    continue
                processed_files += 1
                chunks.extend(self.extract_chunks(pdf_path, target.period_type, target.target_date))
        return BourseRunSummary(
            indexed_chunks=len(chunks),
            processed_files=processed_files,
            errors=errors,
            chunks=chunks,
        ).as_dict()

    async def fetch_resume_pdf(self, target_date: date) -> Path | None:
        target = self._daily_target(target_date)
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            return await self._download_if_needed(client, target)

    async def _download_if_needed(self, client: httpx.AsyncClient, target: PdfTarget) -> Path | None:
        destination = self.cache_dir / target.filename
        if destination.exists():
            return destination
        try:
            response = await client.get(target.url)
            if response.status_code == 200:
                destination.write_bytes(response.content)
                return destination
            return None
        except Exception:
            return None

    def extract_chunks(self, pdf_path: Path, period_type: str = "daily", target_date: date | None = None) -> list[NewsChunk]:
        text_parts: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text_parts.append(text)
        text = "\n".join(text_parts).strip()
        if len(text) < 50:
            return []
        return self.extract_chunks_from_text(text=text, source_key=pdf_path.name, target_date=target_date or date.today(), period_type=period_type)

    def extract_chunks_from_text(self, *, text: str, source_key: str, target_date: date, period_type: str) -> list[NewsChunk]:
        if period_type == "daily":
            return self._extract_daily_chunks(text, source_key, target_date)
        return self._extract_period_chunks(text, source_key, target_date, period_type)

    def _extract_daily_chunks(self, text: str, source_key: str, target_date: date) -> list[NewsChunk]:
        chunks: list[NewsChunk] = []
        market_overview = self._build_market_overview_chunk(text, source_key, target_date)
        if market_overview:
            chunks.append(market_overview)
        sector_chunk = self._build_sector_indices_chunk(text, source_key, target_date)
        if sector_chunk:
            chunks.append(sector_chunk)
        top_movers_chunk = self._build_top_movers_chunk(text, source_key, target_date)
        if top_movers_chunk:
            chunks.append(top_movers_chunk)
        chunks.extend(self._build_stock_performance_chunks(text, source_key, target_date))
        notices_chunk = self._build_notices_chunk(text, source_key, target_date, "corporate_notices")
        if notices_chunk:
            chunks.append(notices_chunk)
        return chunks

    def _extract_period_chunks(self, text: str, source_key: str, target_date: date, period_type: str) -> list[NewsChunk]:
        prefix = {
            "weekly": "weekly",
            "monthly": "monthly",
            "quarterly": "quarterly",
        }[period_type]
        chunks: list[NewsChunk] = []
        summary_text = self._build_period_market_summary_text(text, target_date, period_type)
        if summary_text:
            chunks.append(self._chunk(source_key, f"{prefix}_market_summary", summary_text, target_date))
        stock_summary_text = self._build_period_stock_summary_text(text, target_date, period_type)
        if stock_summary_text:
            chunks.append(self._chunk(source_key, f"{prefix}_stock_performance", stock_summary_text, target_date))
        notices_chunk = self._build_notices_chunk(text, source_key, target_date, f"{prefix}_notices")
        if notices_chunk:
            chunks.append(notices_chunk)
        return chunks

    def _build_market_overview_chunk(self, text: str, source_key: str, target_date: date) -> NewsChunk | None:
        lines = text.splitlines()
        masi_value = self._extract_first_number(MASI_PATTERN, text)
        pct_matches = PERCENT_PATTERN.findall(text)
        daily_pct = pct_matches[0] if len(pct_matches) >= 1 else "0,00"
        ytd_pct = pct_matches[1] if len(pct_matches) >= 2 else "0,00"
        volume_value = self._extract_first_number(VOLUME_PATTERN, text) or "0"
        advancers, decliners = self._count_movers(lines)
        if masi_value is None:
            return None
        chunk_text = (
            f"Bourse de Casablanca — Resume de marche du {target_date.isoformat()}:\n"
            f"  MASI: {masi_value} points\n"
            f"  Variation journaliere: {daily_pct}%\n"
            f"  Performance depuis janvier: {ytd_pct}%\n"
            f"  Volume global: {volume_value} MAD\n"
            f"  {advancers} valeurs en hausse, {decliners} en baisse"
        )
        return self._chunk(source_key, "market_overview", chunk_text, target_date)

    def _build_sector_indices_chunk(self, text: str, source_key: str, target_date: date) -> NewsChunk | None:
        sectors = []
        for line in text.splitlines():
            match = SECTOR_ROW_PATTERN.match(line.strip())
            if not match:
                continue
            sectors.append(
                f"  {match.group('sector')}: {match.group('daily')}% (jour), {match.group('ytd')}% (depuis janvier)"
            )
        if not sectors:
            return None
        chunk_text = f"Bourse de Casablanca — Performances sectorielles du {target_date.isoformat()}:\n" + "\n".join(sectors)
        return self._chunk(source_key, "sector_indices", chunk_text, target_date)

    def _build_top_movers_chunk(self, text: str, source_key: str, target_date: date) -> NewsChunk | None:
        rows = self._extract_stock_rows(text)
        if not rows:
            return None
        sorted_rows = sorted(rows, key=lambda row: row["pct"], reverse=True)
        gainers = sorted_rows[:5]
        losers = sorted(rows, key=lambda row: row["pct"])[:5]
        parts = [f"Bourse de Casablanca — Principales variations du {target_date.isoformat()}:"]
        parts.append("Gagnants:")
        for row in gainers:
            parts.append(f"  {row['name']}: {row['close']:.2f} MAD ({row['pct']:+.2f}%), volume {row['vol']:.0f} MAD")
        parts.append("Perdants:")
        for row in losers:
            parts.append(f"  {row['name']}: {row['close']:.2f} MAD ({row['pct']:+.2f}%), volume {row['vol']:.0f} MAD")
        return self._chunk(source_key, "top_movers", "\n".join(parts), target_date)

    def _build_stock_performance_chunks(self, text: str, source_key: str, target_date: date) -> list[NewsChunk]:
        chunks: list[NewsChunk] = []
        for row in self._extract_stock_rows(text):
            if row["close"] < 1.0 or row["close"] > 30000 or row["vol"] <= 0:
                continue
            chunk_text = (
                f"Bourse de Casablanca — {target_date.isoformat()} — {row['name']}:\n"
                f"  Cours de cloture: {row['close']:.2f} MAD ({row['pct']:+.2f}% vs reference {row['ref']:.2f} MAD)\n"
                f"  Volume journalier: {row['vol']:.0f} MAD, {int(row['qty'])} titres echanges"
            )
            chunks.append(
                self._chunk(
                    source_key,
                    "stock_performance",
                    chunk_text,
                    target_date,
                    metadata={"ticker": row["name"]},
                )
            )
        return chunks

    def _build_notices_chunk(self, text: str, source_key: str, target_date: date, doc_type: str) -> NewsChunk | None:
        notices = self._extract_notices(text)
        if not notices:
            return None
        heading = {
            "corporate_notices": f"Bourse de Casablanca — Avis boursiers (au {target_date.isoformat()}):",
            "weekly_notices": f"Bourse de Casablanca — Avis hebdomadaires (au {target_date.isoformat()}):",
            "monthly_notices": f"Bourse de Casablanca — Avis mensuels (au {target_date.isoformat()}):",
            "quarterly_notices": f"Bourse de Casablanca — Avis trimestriels (au {target_date.isoformat()}):",
        }.get(doc_type, f"Bourse de Casablanca — Avis ({target_date.isoformat()}):")
        lines = [heading] + [f"  [{item['date']}] {item['text']}" for item in notices]
        return self._chunk(source_key, doc_type, "\n".join(lines), target_date)

    def _build_period_market_summary_text(self, text: str, target_date: date, period_type: str) -> str | None:
        masi_value = self._extract_first_number(MASI_PATTERN, text)
        if masi_value is None:
            return None
        pct_matches = PERCENT_PATTERN.findall(text)
        period_pct = pct_matches[0] if pct_matches else "0,00"
        volume_value = self._extract_first_number(VOLUME_PATTERN, text) or "0"
        return (
            f"Bourse de Casablanca — Resume {period_type} du {target_date.isoformat()}:\n"
            f"  MASI: {masi_value} points\n"
            f"  Performance sur la periode: {period_pct}%\n"
            f"  Volume total: {volume_value} MAD"
        )

    def _build_period_stock_summary_text(self, text: str, target_date: date, period_type: str) -> str | None:
        rows = self._extract_stock_rows(text)
        if not rows:
            return None
        lines = [f"## Donnees marche — NOT a news catalyst. Use only to calibrate sentiment_score. ({period_type} {target_date.isoformat()})"]
        for row in rows[:30]:
            lines.append(f"{row['name']}: {row['pct']:+.2f}% sur la periode, cloture {row['close']:.2f} MAD")
        return "\n".join(lines)

    def _extract_stock_rows(self, text: str) -> list[dict]:
        rows: list[dict] = []
        for line in text.splitlines():
            match = STOCK_ROW_PATTERN.match(line.strip())
            if not match:
                continue
            try:
                row = {
                    "name": match.group("name").strip(),
                    "ref": self._parse_fr_number(match.group("ref")),
                    "close": self._parse_fr_number(match.group("close")),
                    "pct": self._parse_fr_number(match.group("pct")),
                    "vol": self._parse_fr_number(match.group("vol")),
                    "qty": self._parse_fr_number(match.group("qty")),
                }
            except ValueError:
                continue
            rows.append(row)
        return rows

    def _extract_notices(self, text: str) -> list[dict]:
        notices: list[dict] = []
        current: dict | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = NOTICE_DATE_PATTERN.match(line)
            if match:
                if current:
                    notices.append(current)
                current = {"date": match.group("date"), "text": match.group("text").strip()}
                continue
            if current and not STOCK_ROW_PATTERN.match(line) and not SECTOR_ROW_PATTERN.match(line):
                current["text"] = f"{current['text']} {line}".strip()
        if current:
            notices.append(current)
        return notices[:50]

    def _count_movers(self, lines: list[str]) -> tuple[int, int]:
        advancers = 0
        decliners = 0
        for row in self._extract_stock_rows("\n".join(lines)):
            if row["pct"] > 0:
                advancers += 1
            elif row["pct"] < 0:
                decliners += 1
        return advancers, decliners

    def _extract_first_number(self, pattern: re.Pattern, text: str) -> str | None:
        match = pattern.search(text)
        if not match:
            return None
        value = match.group("value")
        return self._normalize_fr_number_text(value)

    def _parse_fr_number(self, value: str) -> float:
        normalized = self._normalize_fr_number_text(value)
        return float(normalized)

    def _normalize_fr_number_text(self, value: str) -> str:
        cleaned = value.replace("\xa0", "").replace(" ", "").replace(".", "").replace(",", ".")
        return cleaned

    def _chunk(self, source_key: str, doc_type: str, text: str, target_date: date, metadata: dict | None = None) -> NewsChunk:
        meta = {"doc_type": doc_type, **(metadata or {})}
        chunk_id = md5(f"{source_key}|{doc_type}|{text[:120]}".encode("utf-8")).hexdigest()
        return NewsChunk(
            chunk_id=chunk_id,
            text=text,
            source="Casablanca Bourse PDF",
            published_at=datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc),
            similarity_score=1.0,
            url=None,
            metadata=meta,
        )

    def last_completed_trading_days(self, count: int = 10) -> list[date]:
        days: list[date] = []
        current = date.today() - timedelta(days=1)
        while len(days) < count:
            if current.weekday() < 5:
                days.append(current)
            current -= timedelta(days=1)
        return days

    def _daily_target(self, target_date: date) -> PdfTarget:
        filename = f"resume_seance_{target_date:%Y%m%d}.pdf"
        url = f"https://media.casablanca-bourse.com/sites/default/files/es-auto-upload/fr/{filename}"
        return PdfTarget("daily", target_date, url, filename)

    def _daily_targets(self, count: int) -> list[PdfTarget]:
        return [self._daily_target(target_date) for target_date in self.last_completed_trading_days(count)]

    def _weekly_targets(self, count: int) -> list[PdfTarget]:
        targets: list[PdfTarget] = []
        current = date.today() - timedelta(days=1)
        while current.weekday() != 4:
            current -= timedelta(days=1)
        while len(targets) < count:
            month_dir = current.strftime("%Y-%m")
            filename = f"resume_hebdo_{current:%Y%m%d}.pdf"
            url = f"https://media.casablanca-bourse.com/sites/default/files/{month_dir}/{filename}"
            targets.append(PdfTarget("weekly", current, url, filename))
            current -= timedelta(days=7)
        return targets

    def _monthly_targets(self, count: int) -> list[PdfTarget]:
        targets: list[PdfTarget] = []
        ref = date.today().replace(day=1) - timedelta(days=1)
        while len(targets) < count:
            month_end = self._last_trading_day_of_month(ref.year, ref.month)
            month_dir = month_end.strftime("%Y-%m")
            filename = f"resume_mensuel_{month_end:%Y%m%d}.pdf"
            url = f"https://media.casablanca-bourse.com/sites/default/files/{month_dir}/{filename}"
            targets.append(PdfTarget("monthly", month_end, url, filename))
            ref = ref.replace(day=1) - timedelta(days=1)
        return targets

    def _quarterly_targets(self, count: int) -> list[PdfTarget]:
        quarter_months = (3, 6, 9, 12)
        targets: list[PdfTarget] = []
        current = date.today().replace(day=1) - timedelta(days=1)
        while len(targets) < count:
            while current.month not in quarter_months:
                current = current.replace(day=1) - timedelta(days=1)
            quarter_end = self._last_trading_day_of_month(current.year, current.month)
            month_dir = quarter_end.strftime("%Y-%m")
            filename = f"resume_trimestriel_{quarter_end:%Y%m%d}.pdf"
            url = f"https://media.casablanca-bourse.com/sites/default/files/{month_dir}/{filename}"
            targets.append(PdfTarget("quarterly", quarter_end, url, filename))
            current = current.replace(day=1) - timedelta(days=1)
        return targets

    def _last_trading_day_of_month(self, year: int, month: int) -> date:
        last_day = calendar.monthrange(year, month)[1]
        current = date(year, month, last_day)
        while current.weekday() >= 5:
            current -= timedelta(days=1)
        return current
