from __future__ import annotations

from typing import Any

from trading_agents.core.models import NewsChunk

try:
    from langsmith import trace

    LANGSMITH_TRACE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    trace = None
    LANGSMITH_TRACE_AVAILABLE = False


class LangSmithRetrievalTracer:
    def __init__(self, *, enabled: bool, project_name: str):
        self.enabled = enabled and LANGSMITH_TRACE_AVAILABLE
        self.project_name = project_name

    def log_search(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        collections: list[str],
        results: list[NewsChunk],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or trace is None:
            return
        clean_metadata = self._clean_value(metadata or {})
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": self._clean_value(filters),
            "collections": collections,
        }
        outputs = {
            "result_count": len(results),
            "results": [
                {
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "url": item.url,
                    "doc_type": item.metadata.get("doc_type"),
                    "ticker": item.metadata.get("ticker"),
                    "similarity_score": item.similarity_score,
                    "low_confidence": item.low_confidence,
                }
                for item in results
            ],
        }
        with trace(
            "news_retrieval",
            run_type="retriever",
            project_name=self.project_name,
            inputs=inputs,
            metadata=clean_metadata,
            tags=["retrieval", "news", "morocco-trading-agents"],
        ) as run_tree:
            run_tree.end(outputs=outputs)

    def _clean_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._clean_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._clean_value(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._clean_value(value.model_dump(mode="json"))
        return str(value)
