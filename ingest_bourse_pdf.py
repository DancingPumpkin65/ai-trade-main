from __future__ import annotations

import asyncio

from trading_agents.core.config import get_settings
from trading_agents.core.data.bourse_fetcher import BourseDataFetcher


async def main() -> None:
    settings = get_settings()
    fetcher = BourseDataFetcher(settings.data_dir / "bourse_pdfs")
    summary = await fetcher.run_daily()
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
