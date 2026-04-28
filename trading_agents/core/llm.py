from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class OllamaAgentLLM:
    def __init__(self, *, base_url: str, model: str, enabled: bool = False, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.timeout = timeout

    def generate_structured(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        context: dict[str, Any],
        response_model: type[BaseModel],
    ) -> BaseModel | None:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "prompt": (
                f"You are the {agent_name} agent inside a Morocco trading system.\n"
                f"{system_prompt}\n\n"
                f"Return JSON only that matches this schema exactly:\n"
                f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}\n\n"
                f"Context JSON:\n{json.dumps(context, ensure_ascii=False, default=_json_default)}"
            ),
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            if not raw:
                return None
            return response_model.model_validate_json(raw)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return None


_default_agent_llm: OllamaAgentLLM | None = None


def set_default_agent_llm(client: OllamaAgentLLM | None) -> None:
    global _default_agent_llm
    _default_agent_llm = client


def get_default_agent_llm() -> OllamaAgentLLM | None:
    return _default_agent_llm
