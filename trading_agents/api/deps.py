from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trading_agents.core.config import get_settings
from trading_agents.core.services import AppServices


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    return AppServices(get_settings())


ServicesDep = Annotated[AppServices, Depends(get_services)]

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_username(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    services: ServicesDep,
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return services.auth_service.authenticate_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUsernameDep = Annotated[str, Depends(get_current_username)]
