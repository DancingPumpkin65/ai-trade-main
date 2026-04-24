from __future__ import annotations

from fastapi import APIRouter

from trading_agents.api.deps import get_services


router = APIRouter(tags=["history"])


@router.get("/history")
def history():
    services = get_services()
    return [record.model_dump(mode="json") for record in services.history()]
