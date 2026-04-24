from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from trading_agents.api.deps import get_services
from trading_agents.core.models import GenerateSignalRequest


router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("/generate")
def generate(payload: GenerateSignalRequest):
    services = get_services()
    try:
        return services.generate(payload).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/generate/stream")
def stream(request_id: str = Query(...)):
    services = get_services()
    events = services.stream_events(request_id)

    def event_generator():
        for event in events:
            time.sleep(0.01)
            yield {
                "event": event["event_type"],
                "data": json.dumps(event["payload"]),
            }

    return EventSourceResponse(event_generator())


@router.get("/{request_id}")
def get_signal(request_id: str):
    services = get_services()
    try:
        return services.export_signal_detail(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{request_id}/approve")
def approve(request_id: str):
    services = get_services()
    try:
        return services.approve(request_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{request_id}/reject")
def reject(request_id: str):
    services = get_services()
    try:
        return services.reject(request_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{request_id}/alpaca-order")
def get_alpaca_order(request_id: str):
    services = get_services()
    try:
        return services.get_alpaca_order(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
