from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


ToolHandler = Callable[..., Any]


class MCPServer:
    def __init__(self):
        self._tools: dict[str, dict[str, ToolHandler]] = defaultdict(dict)

    def register_tool(self, namespace: str, name: str, handler: ToolHandler) -> None:
        self._tools[namespace][name] = handler

    def list_tools(self, namespace: str) -> list[str]:
        return sorted(self._tools.get(namespace, {}).keys())

    def call_tool(self, namespace: str, name: str, **kwargs):
        if namespace not in self._tools or name not in self._tools[namespace]:
            raise KeyError(f"Tool not registered: {namespace}.{name}")
        return self._tools[namespace][name](**kwargs)
