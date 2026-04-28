from __future__ import annotations

from fastapi import APIRouter, Depends

from trading_agents.api.deps import ServicesDep, get_current_username


router = APIRouter(tags=["history"], dependencies=[Depends(get_current_username)])


@router.get("/history")
def history(services: ServicesDep):
    return [record.model_dump(mode="json") for record in services.history()]
