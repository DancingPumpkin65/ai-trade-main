from __future__ import annotations

from functools import lru_cache

from trading_agents.core.config import get_settings
from trading_agents.core.services import AppServices


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    return AppServices(get_settings())
