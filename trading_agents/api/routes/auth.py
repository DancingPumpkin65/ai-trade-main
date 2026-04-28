from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trading_agents.api.deps import ServicesDep
from trading_agents.core.models import UserCreate, UserLogin


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(payload: UserCreate, services: ServicesDep):
    try:
        return services.auth_service.register(payload.username, payload.password)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login")
def login(payload: UserLogin, services: ServicesDep):
    try:
        return services.auth_service.login(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
