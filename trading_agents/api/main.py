from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from trading_agents.api.routes import auth, history, signals
from trading_agents.api.deps import get_services


@asynccontextmanager
async def lifespan(_: FastAPI):
    services = get_services()
    try:
        yield
    finally:
        services.close()


app = FastAPI(title="Morocco Trading Agents", lifespan=lifespan)


@app.get("/health")
def health():
    return get_services().health()


app.include_router(auth.router)
app.include_router(history.router)
app.include_router(signals.router)
