from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from trading_agents.api.deps import ServicesDep, get_current_username
from trading_agents.core.models import GenerateSignalRequest, RiskPreference, SignalStatus, TimeHorizon


router = APIRouter(
    prefix="/signals",
    tags=["signals"],
    dependencies=[Depends(get_current_username)],
)


@router.post("/generate")
def generate(payload: GenerateSignalRequest, services: ServicesDep):
    try:
        return services.generate(payload).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/generate/stream")
def stream(
    services: ServicesDep,
    request_id: str | None = Query(default=None),
    symbol: str | None = Query(default=None),
    capital: float | None = Query(default=None),
    prompt: str | None = Query(default=None),
    risk_profile: RiskPreference | None = Query(default=None),
    time_horizon: TimeHorizon | None = Query(default=None),
):
    created_live_request = False

    if request_id is None:
        payload = GenerateSignalRequest(
            symbol=symbol,
            capital=capital,
            prompt=prompt,
            risk_profile=risk_profile,
            time_horizon=time_horizon,
        )
        try:
            generated = services.generate_live(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        request_id = generated.request_id
        created_live_request = True

    terminal_statuses = {
        SignalStatus.COMPLETED.value,
        SignalStatus.FAILED.value,
        SignalStatus.APPROVED.value,
        SignalStatus.REJECTED.value,
    }

    def event_generator():
        last_event_id = 0
        if created_live_request:
            yield {
                "event": "request_started",
                "data": json.dumps({"request_id": request_id, "status": SignalStatus.RUNNING.value}),
            }

        while True:
            events = services.stream_events_after(request_id, last_event_id)
            if events:
                for event in events:
                    last_event_id = event["id"]
                    yield {
                        "event": event["event_type"],
                        "data": json.dumps(event["payload"]),
                    }

            try:
                record = services.get_signal(request_id)
            except ValueError:
                break

            if record.status.value in terminal_statuses and not events:
                break

            time.sleep(0.05)

    return EventSourceResponse(event_generator())


@router.get("/{request_id}")
def get_signal(request_id: str, services: ServicesDep):
    try:
        return services.export_signal_detail(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{request_id}/approve")
def approve(request_id: str, services: ServicesDep):
    try:
        return services.approve(request_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{request_id}/reject")
def reject(request_id: str, services: ServicesDep):
    try:
        return services.reject(request_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{request_id}/alpaca-order")
def get_alpaca_order(request_id: str, services: ServicesDep):
    try:
        return services.get_alpaca_order(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{request_id}/opportunities/{symbol}/alpaca-order")
def get_opportunity_alpaca_order(request_id: str, symbol: str, services: ServicesDep):
    try:
        return services.get_universe_opportunity_alpaca_order(request_id, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{request_id}/opportunities/{symbol}/approve")
def approve_opportunity(request_id: str, symbol: str, services: ServicesDep):
    try:
        return services.approve_universe_opportunity(request_id, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{request_id}/opportunities/{symbol}/reject")
def reject_opportunity(request_id: str, symbol: str, services: ServicesDep):
    try:
        return services.reject_universe_opportunity(request_id, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
