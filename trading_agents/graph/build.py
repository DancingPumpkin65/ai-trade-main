from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from trading_agents.core.broker.alpaca import AlpacaPreviewService
from trading_agents.core.data.bourse_fetcher import BourseDataFetcher
from trading_agents.core.data.drahmi import DrahmiClient
from trading_agents.core.data.news_global import MarketAuxClient
from trading_agents.core.data.news_morocco import MoroccoNewsClient
from trading_agents.core.intent.policy import IntentPolicyEngine
from trading_agents.core.mcp.server import MCPServer
from trading_agents.core.models import (
    AlpacaOrderStatus,
    CoordinatorOutput,
    NewsChunk,
    RankedCandidate,
    RequestIntent,
    RequestMode,
    RiskOutput,
    SignalStatus,
    SentimentOutput,
    StockInfo,
    TechnicalOutput,
    TradeOpportunity,
    TradeOpportunityList,
    TradingSignal,
)
from trading_agents.core.observability import LangSmithRetrievalTracer
from trading_agents.core.rag.indexer import Indexer
from trading_agents.core.rag.retriever import NewsRetriever
from trading_agents.core.rag.store import build_vector_store
from trading_agents.core.storage import Storage
from trading_agents.graph.coordinator_node import run_coordinator_agent
from trading_agents.graph.enforce_limits import enforce_limits
from trading_agents.graph.helpers import analyze_technical_features, calculate_position_size
from trading_agents.graph.risk_node import run_risk_agent
from trading_agents.graph.sentiment_node import run_sentiment_agent
from trading_agents.graph.state import GraphState
from trading_agents.graph.technical_node import run_technical_agent

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command, interrupt

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when integrations are missing
    LANGGRAPH_AVAILABLE = False
    END = START = StateGraph = InMemorySaver = Command = interrupt = None

try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    SQLITE_SAVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when sqlite saver is missing
    SQLITE_SAVER_AVAILABLE = False
    SqliteSaver = None


class TradingGraphService:
    def __init__(
        self,
        *,
        storage: Storage,
        drahmi_client: DrahmiClient,
        morocco_news_client: MoroccoNewsClient,
        marketaux_client: MarketAuxClient,
        alpaca_preview_service: AlpacaPreviewService,
        checkpoint_path: Path | None = None,
        chroma_persist_dir: Path | None = None,
        bourse_cache_dir: Path | None = None,
        env: str = "dev",
        langsmith_tracing: bool = False,
        langsmith_project: str = "morocco-trading-agents",
        max_agent_iterations: int = 8,
    ):
        self.storage = storage
        self.drahmi_client = drahmi_client
        self.morocco_news_client = morocco_news_client
        self.marketaux_client = marketaux_client
        self.alpaca_preview_service = alpaca_preview_service
        self.checkpoint_path = checkpoint_path
        self.chroma_persist_dir = chroma_persist_dir
        self.bourse_cache_dir = bourse_cache_dir
        self.env = env
        self.max_agent_iterations = max_agent_iterations
        self.policy_engine = IntentPolicyEngine()
        self.retrieval_tracer = LangSmithRetrievalTracer(
            enabled=langsmith_tracing,
            project_name=langsmith_project,
        )
        self.vector_store = build_vector_store(
            persist_dir=chroma_persist_dir or Path("./data/chroma"),
            env=env,
            prefer_chroma=True,
        )
        self.indexer = Indexer(self.vector_store)
        self.retriever = NewsRetriever(self.vector_store, tracer=self.retrieval_tracer)
        self.bourse_fetcher = BourseDataFetcher(bourse_cache_dir or Path("./data/bourse_pdfs"))
        self._last_bourse_ingestion_date: date | None = None
        self._issuer_ingestion_cache: set[tuple[date, str]] = set()
        self.mcp = MCPServer()
        self._active_state_context: dict[str, Any] = {}
        self._register_tools()

        self.langgraph_enabled = LANGGRAPH_AVAILABLE
        self._checkpointer_context = None
        self.checkpointer = None
        self.single_symbol_graph = None
        if self.langgraph_enabled:
            self._initialize_langgraph_runtime()

    def close(self) -> None:
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(None, None, None)
            self._checkpointer_context = None

    def _initialize_langgraph_runtime(self) -> None:
        if SQLITE_SAVER_AVAILABLE and self.checkpoint_path is not None:
            self._checkpointer_context = SqliteSaver.from_conn_string(str(self.checkpoint_path))
            self.checkpointer = self._checkpointer_context.__enter__()
        else:
            self.checkpointer = InMemorySaver()
        self.single_symbol_graph = self._build_single_symbol_graph()

    def _build_single_symbol_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("prepare_context", self._prepare_context_node)
        builder.add_node("sentiment_agent", self._sentiment_node)
        builder.add_node("technical_agent", self._technical_node)
        builder.add_node("risk_agent", self._risk_node)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("coordinator_agent", self._coordinator_node)
        builder.add_node("enforce_limits", self._enforce_limits_node)

        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "sentiment_agent")
        builder.add_edge("prepare_context", "technical_agent")
        builder.add_edge("sentiment_agent", "risk_agent")
        builder.add_edge("technical_agent", "risk_agent")
        builder.add_conditional_edges(
            "risk_agent",
            self._risk_router,
            {
                "human_review": "human_review",
                "coordinator_agent": "coordinator_agent",
            },
        )
        builder.add_conditional_edges(
            "human_review",
            self._human_review_router,
            {
                "coordinator_agent": "coordinator_agent",
                "end": END,
            },
        )
        builder.add_edge("coordinator_agent", "enforce_limits")
        builder.add_edge("enforce_limits", END)
        return builder.compile(checkpointer=self.checkpointer, name="morocco-trading-single-symbol")

    def _register_tools(self) -> None:
        self.mcp.register_tool("sentiment", "search_news", lambda query, top_k=5, filters=None: self.retriever.search_news(query, top_k, filters, metadata=self._retrieval_metadata(query_source="sentiment_tool", top_k=top_k)))
        self.mcp.register_tool("sentiment", "get_market_data", lambda symbol: asyncio.run(self.drahmi_client.get_stock(symbol)))
        self.mcp.register_tool("technical", "get_market_data", lambda symbol: asyncio.run(self.drahmi_client.get_stock(symbol)))
        self.mcp.register_tool("technical", "analyze_technical", lambda symbol: analyze_technical_features(asyncio.run(self.drahmi_client.get_stock(symbol))))
        self.mcp.register_tool("risk", "get_sentiment_output", lambda: self._active_state_context.get("sentiment_output"))
        self.mcp.register_tool("risk", "get_technical_output", lambda: self._active_state_context.get("technical_output"))
        self.mcp.register_tool("risk", "calculate_size", self._calculate_size_from_active_context)
        self.mcp.register_tool(
            "coordinator",
            "get_all_outputs",
            lambda: {
                "sentiment": self._active_state_context.get("sentiment_output"),
                "technical": self._active_state_context.get("technical_output"),
                "risk": self._active_state_context.get("risk_output"),
            },
        )

    def _calculate_size_from_active_context(self, symbol: str, action: str, capital: float):
        technical_features = self._active_state_context.get("technical_features") or {}
        request_intent = self._active_state_context.get("request_intent")
        conservative_posture = bool(request_intent and request_intent.risk_preference.value == "CONSERVATIVE")
        sizing = calculate_position_size(
            symbol=symbol,
            action=action,
            capital=capital,
            volatility_estimate=float(technical_features.get("annualized_volatility", 0.2)),
            is_fixing_mode=bool(technical_features.get("is_fixing_mode")),
            conservative_posture=conservative_posture,
        )
        return sizing

    def _record_tool_interaction(self, request_id: str, agent: str, tool_name: str, args: dict[str, Any], result: Any) -> None:
        self.storage.add_event(request_id, "tool_call", {"agent": agent, "tool": tool_name, "args": args})
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        self.storage.add_event(request_id, "tool_result", {"agent": agent, "tool": tool_name, "result": payload})

    def _retrieval_metadata(self, *, query_source: str, top_k: int) -> dict[str, Any]:
        request_intent = self._active_state_context.get("request_intent")
        return {
            "request_id": self._active_state_context.get("request_id"),
            "symbol": self._active_state_context.get("symbol"),
            "query_source": query_source,
            "top_k": top_k,
            "request_mode": request_intent.request_mode.value if request_intent else None,
            "time_horizon": request_intent.time_horizon.value if request_intent else None,
            "risk_preference": request_intent.risk_preference.value if request_intent else None,
        }

    def _safe_sentiment_output(self, symbol: str, note: str) -> SentimentOutput:
        return SentimentOutput(
            sentiment_score=0.5,
            catalysts=[],
            cited_article_ids=[],
            confidence=0.2,
            rationale_fr=f"Analyse de sentiment prudente pour {symbol}. {note}",
        )

    def _safe_risk_output(self, request_intent: RequestIntent, technical_output: TechnicalOutput) -> RiskOutput:
        return RiskOutput(
            action="HOLD",
            position_size_pct=0.0,
            position_value_mad=0.0,
            stop_loss_pct=min(0.02, 0.06 if technical_output.volatility_estimate > 0 else 0.10),
            take_profit_pct=0.03,
            risk_score=0.2,
            volatility_estimate=technical_output.volatility_estimate,
            rationale=f"Fallback safe output. Risk preference: {request_intent.risk_preference.value}.",
        )

    def _safe_coordinator_output(self, symbol: str, request_intent: RequestIntent) -> CoordinatorOutput:
        return CoordinatorOutput(
            action="HOLD",
            position_size_pct=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            risk_score=0.2,
            rationale_fr=f"Demande comprise: {request_intent.operator_visible_note_fr} Conclusion système sur {symbol}: HOLD par défaut de sécurité.",
            dissenting_views=[],
            confidence=0.2,
        )

    async def ingest_news(self, symbol: str | None = None) -> None:
        await self.ingest_bourse_documents()
        morocco_news = await self.morocco_news_client.fetch()
        self.indexer.upsert_news(morocco_news)
        if symbol:
            global_news = await self.marketaux_client.fetch_for_symbol(symbol)
            self.indexer.upsert_news(global_news)
            await self.ingest_issuer_publications(symbol)

    async def ingest_bourse_documents(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self.env == "test" or self._last_bourse_ingestion_date == today:
            return
        summary = await self.bourse_fetcher.run_daily()
        chunks = summary.get("chunks", [])
        if chunks:
            self.indexer.upsert_macro_documents(chunks)
            self.storage.add_event(
                "system",
                "bourse_ingestion",
                {
                    "indexed_chunks": summary.get("indexed_chunks", 0),
                    "processed_files": summary.get("processed_files", 0),
                },
            )
        self._last_bourse_ingestion_date = today

    async def ingest_issuer_publications(self, symbol: str) -> None:
        today = datetime.now(timezone.utc).date()
        cache_key = (today, symbol.upper())
        if cache_key in self._issuer_ingestion_cache:
            return
        stock = await self.drahmi_client.get_stock(symbol)
        chunks = await self.bourse_fetcher.fetch_issuer_publication_chunks([stock], limit_pdfs=3)
        if chunks:
            self.indexer.upsert_macro_documents(chunks)
            self.storage.add_event(
                "system",
                "issuer_publication_ingestion",
                {
                    "symbol": stock.symbol,
                    "indexed_chunks": len(chunks),
                },
            )
        self._issuer_ingestion_cache.add(cache_key)

    def start(self, intent: RequestIntent) -> None:
        self.storage.update_request(intent.request_id, status=SignalStatus.RUNNING)
        self.storage.add_event(intent.request_id, "pipeline_start", {"request_id": intent.request_id})
        if intent.request_mode == RequestMode.UNIVERSE_SCAN:
            opportunity_list = self._run_universe_scan(intent)
            self.storage.update_request(
                intent.request_id,
                status=SignalStatus.COMPLETED,
                opportunity_list=opportunity_list,
                human_review_required=False,
            )
            self.storage.add_event(intent.request_id, "pipeline_complete", {"mode": "UNIVERSE_SCAN"})
            return

        if self.single_symbol_graph is not None:
            self._start_with_langgraph(intent)
            return

        result = self._run_symbol_legacy(intent, intent.symbols_requested[0])
        self._persist_legacy_result(intent.request_id, result)

    def resume(self, request_id: str, decision: str) -> None:
        if self.single_symbol_graph is not None:
            self._resume_with_langgraph(request_id, decision)
            return

        saved_state = self.storage.get_saved_state(request_id)
        if not saved_state:
            raise ValueError("No paused state exists for this request.")
        if decision == "rejected":
            self.storage.update_request(request_id, status=SignalStatus.REJECTED, human_review_required=False)
            self.storage.add_event(request_id, "human_review_rejected", {})
            return

        coordinator_output = CoordinatorOutput.model_validate(saved_state["coordinator_output"])
        final_signal = enforce_limits(
            symbol=saved_state["symbol"],
            request_id=request_id,
            coordinator_output=coordinator_output,
            is_fixing_mode=saved_state["technical_features"]["is_fixing_mode"],
            capital=saved_state["capital"],
        )
        alpaca_preview = self.alpaca_preview_service.prepare_preview(final_signal)
        self.storage.update_request(
            request_id,
            status=SignalStatus.COMPLETED,
            human_review_required=False,
            final_signal=final_signal,
            coordinator_output=coordinator_output,
            alpaca_order=alpaca_preview,
            alpaca_order_status=alpaca_preview.status,
            state={},
        )
        self.storage.add_event(request_id, "human_review_approved", {"symbol": saved_state["symbol"]})
        self.storage.add_event(request_id, "alpaca_preview_prepared", alpaca_preview.model_dump(mode="json"))

    def _start_with_langgraph(self, intent: RequestIntent) -> None:
        symbol = intent.symbols_requested[0]
        config = self._graph_config(intent.request_id)
        initial_state = self._initial_graph_state(intent, symbol)
        result = self.single_symbol_graph.invoke(initial_state, config=config)
        if "__interrupt__" in result:
            snapshot = self.single_symbol_graph.get_state(config)
            state_values = dict(snapshot.values)
            self.storage.update_request(
                intent.request_id,
                status=SignalStatus.WAITING_HUMAN,
                human_review_required=True,
                errors=state_values.get("errors", []),
                state=state_values,
            )
            self.storage.add_event(
                intent.request_id,
                "human_review_required",
                {
                    "symbol": symbol,
                    "interrupts": [getattr(item, "value", None) for item in result.get("__interrupt__", [])],
                },
            )
            return

        self._persist_langgraph_completion(intent.request_id, result, human_review_required=False)

    def _resume_with_langgraph(self, request_id: str, decision: str) -> None:
        config = self._graph_config(request_id)
        result = self.single_symbol_graph.invoke(Command(resume=decision), config=config)
        snapshot = self.single_symbol_graph.get_state(config)
        state_values = dict(snapshot.values)
        if decision == "rejected" or state_values.get("human_review_decision") == "rejected":
            self.storage.update_request(
                request_id,
                status=SignalStatus.REJECTED,
                human_review_required=False,
                errors=state_values.get("errors", []),
                state=state_values,
            )
            self.storage.add_event(request_id, "human_review_rejected", {})
            return

        self.storage.add_event(request_id, "human_review_approved", {"symbol": state_values.get("symbol")})
        self._persist_langgraph_completion(request_id, result, human_review_required=False, prepare_alpaca_preview=True)

    def _persist_langgraph_completion(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        human_review_required: bool,
        prepare_alpaca_preview: bool = False,
    ) -> None:
        final_signal = self._coerce_model(result.get("final_signal"), TradingSignal)
        coordinator_output = self._coerce_model(result.get("coordinator_output"), CoordinatorOutput)
        alpaca_preview = None
        alpaca_status = None
        if prepare_alpaca_preview and final_signal is not None:
            alpaca_preview = self.alpaca_preview_service.prepare_preview(final_signal)
            alpaca_status = alpaca_preview.status
            self.storage.add_event(request_id, "alpaca_preview_prepared", alpaca_preview.model_dump(mode="json"))

        self.storage.update_request(
            request_id,
            status=SignalStatus.COMPLETED,
            human_review_required=human_review_required,
            final_signal=final_signal,
            coordinator_output=coordinator_output,
            alpaca_order=alpaca_preview,
            alpaca_order_status=alpaca_status,
            errors=result.get("errors", []),
            state=result,
        )
        self.storage.add_event(request_id, "pipeline_complete", {"symbol": final_signal.symbol if final_signal else result.get("symbol")})

    def _persist_legacy_result(self, request_id: str, result: dict[str, Any]) -> None:
        if result["human_review_required"]:
            self.storage.update_request(
                request_id,
                status=SignalStatus.WAITING_HUMAN,
                human_review_required=True,
                coordinator_output=result.get("coordinator_output"),
                errors=result["errors"],
                state=result,
            )
            self.storage.add_event(request_id, "human_review_required", {"symbol": result["symbol"]})
            return

        final_signal = result["final_signal"]
        coordinator_output = result["coordinator_output"]
        self.storage.update_request(
            request_id,
            status=SignalStatus.COMPLETED,
            human_review_required=False,
            final_signal=final_signal,
            coordinator_output=coordinator_output,
            errors=result["errors"],
        )
        self.storage.add_event(request_id, "pipeline_complete", {"symbol": final_signal.symbol})

    def _execute_technical_with_ground_truth(self, *, request_id: str, symbol: str, stock: StockInfo, initial_retry_count: int = 0) -> tuple[TechnicalOutput, dict[str, Any], int, list[str]]:
        retry_count = initial_retry_count
        mismatch_feedback: str | None = None
        errors: list[str] = []
        while True:
            technical_output, technical_features = run_technical_agent(stock, mismatch_feedback=mismatch_feedback)
            ground_truth_bias = technical_features["directional_bias"]
            if technical_output.directional_bias == ground_truth_bias:
                return technical_output, technical_features, retry_count, errors

            retry_count += 1
            mismatch_message = (
                f"Technical bias mismatch for {symbol}: model={technical_output.directional_bias}, "
                f"ground_truth={ground_truth_bias}, retry={retry_count}."
            )
            errors.append(mismatch_message)
            self.storage.add_event(
                request_id,
                "technical_retry",
                {
                    "symbol": symbol,
                    "retry_count": retry_count,
                    "model_bias": technical_output.directional_bias,
                    "ground_truth_bias": ground_truth_bias,
                },
            )
            if retry_count >= 2:
                corrected_output = technical_output.model_copy(
                    update={
                        "directional_bias": ground_truth_bias,
                        "trend_summary": f"{technical_output.trend_summary} Correction finale appliquée selon la vérité terrain Python.",
                    }
                )
                return corrected_output, technical_features, retry_count, errors

            mismatch_feedback = (
                f"Votre biais précédent ({technical_output.directional_bias}) ne correspond pas à la vérité terrain "
                f"calculée en Python ({ground_truth_bias}). Réévaluez uniquement la narration, pas les nombres."
            )

    def _run_sentiment_loop(
        self,
        *,
        request_id: str,
        symbol: str,
        request_intent: RequestIntent,
        scratchpad: list[dict],
    ) -> tuple[SentimentOutput, list[dict], list[str]]:
        messages = list(scratchpad)
        errors: list[str] = []
        market_data = None
        news_chunks: list[NewsChunk] = []
        searches = 0
        for iteration in range(1, self.max_agent_iterations + 1):
            if market_data is None:
                args = {"symbol": symbol}
                market_data = self.mcp.call_tool("sentiment", "get_market_data", **args)
                self._record_tool_interaction(request_id, "sentiment", "get_market_data", args, market_data)
                messages.append({"role": "tool", "tool": "get_market_data", "iteration": iteration, "content": f"Loaded market data for {symbol}."})
                continue
            if searches == 0:
                query = f"{symbol} Morocco"
                args = {"query": query, "top_k": 5, "filters": None}
                news_chunks = self.mcp.call_tool("sentiment", "search_news", **args)
                searches += 1
                self._record_tool_interaction(request_id, "sentiment", "search_news", args, [chunk.model_dump(mode="json") for chunk in news_chunks])
                messages.append({"role": "assistant", "iteration": iteration, "content": f"Searching news with query: {query}"})
                continue
            high_confidence = [chunk for chunk in news_chunks if chunk.similarity_score >= 0.55]
            if searches < 2 and len(high_confidence) < 2:
                query = f"{symbol} corporate notice Morocco"
                args = {"query": query, "top_k": 5, "filters": None}
                news_chunks = self.mcp.call_tool("sentiment", "search_news", **args)
                searches += 1
                self._record_tool_interaction(request_id, "sentiment", "search_news", args, [chunk.model_dump(mode="json") for chunk in news_chunks])
                messages.append({"role": "assistant", "iteration": iteration, "content": f"Retrying retrieval with query: {query}"})
                continue
            output = run_sentiment_agent(
                symbol=symbol,
                request_intent=request_intent,
                market_data=market_data,
                news_chunks=news_chunks,
            )
            messages.append({"role": "assistant", "iteration": iteration, "content": "Sentiment synthesis completed."})
            return output, messages, errors
        errors.append(f"Sentiment agent exceeded the {self.max_agent_iterations}-iteration cap.")
        messages.append({"role": "system", "content": errors[-1]})
        return self._safe_sentiment_output(symbol, "Le moteur a atteint la limite d'itérations."), messages, errors

    def _run_technical_loop(
        self,
        *,
        request_id: str,
        symbol: str,
        stock: StockInfo,
        scratchpad: list[dict],
        initial_retry_count: int = 0,
    ) -> tuple[TechnicalOutput, dict[str, Any], int, list[dict], list[str]]:
        messages = list(scratchpad)
        errors: list[str] = []
        fetched_market_data = None
        technical_features = None
        technical_output = None
        retry_count = initial_retry_count
        for iteration in range(1, self.max_agent_iterations + 1):
            if fetched_market_data is None:
                args = {"symbol": symbol}
                fetched_market_data = self.mcp.call_tool("technical", "get_market_data", **args)
                self._record_tool_interaction(request_id, "technical", "get_market_data", args, fetched_market_data)
                messages.append({"role": "tool", "tool": "get_market_data", "iteration": iteration, "content": f"Fetched technical market context for {symbol}."})
                continue
            if technical_features is None:
                args = {"symbol": symbol}
                features = self.mcp.call_tool("technical", "analyze_technical", **args)
                technical_features = features.model_dump(mode="json") if hasattr(features, "model_dump") else features
                self._record_tool_interaction(request_id, "technical", "analyze_technical", args, technical_features)
                technical_output, technical_features, retry_count, retry_errors = self._execute_technical_with_ground_truth(
                    request_id=request_id,
                    symbol=symbol,
                    stock=stock,
                    initial_retry_count=retry_count,
                )
                errors.extend(retry_errors)
                messages.append({"role": "assistant", "iteration": iteration, "content": "Technical synthesis completed."})
                return technical_output, technical_features, retry_count, messages, errors
        errors.append(f"Technical agent exceeded the {self.max_agent_iterations}-iteration cap.")
        messages.append({"role": "system", "content": errors[-1]})
        technical_output, technical_features = run_technical_agent(stock, mismatch_feedback="Fallback after iteration cap.")
        return technical_output, technical_features, retry_count, messages, errors

    def _run_risk_loop(
        self,
        *,
        request_id: str,
        symbol: str,
        capital: float,
        request_intent: RequestIntent,
        scratchpad: list[dict],
    ) -> tuple[RiskOutput, list[dict], list[str]]:
        messages = list(scratchpad)
        errors: list[str] = []
        sentiment_output = None
        technical_output = None
        sizing = None
        for iteration in range(1, self.max_agent_iterations + 1):
            if sentiment_output is None:
                sentiment_output = self.mcp.call_tool("risk", "get_sentiment_output")
                self._record_tool_interaction(request_id, "risk", "get_sentiment_output", {}, sentiment_output)
                messages.append({"role": "tool", "tool": "get_sentiment_output", "iteration": iteration, "content": "Loaded upstream sentiment output."})
                continue
            if technical_output is None:
                technical_output = self.mcp.call_tool("risk", "get_technical_output")
                self._record_tool_interaction(request_id, "risk", "get_technical_output", {}, technical_output)
                messages.append({"role": "tool", "tool": "get_technical_output", "iteration": iteration, "content": "Loaded upstream technical output."})
                continue
            sentiment_model = SentimentOutput.model_validate(sentiment_output)
            technical_model = TechnicalOutput.model_validate(technical_output)
            action = (
                "BUY"
                if technical_model.directional_bias == "BULLISH" and sentiment_model.sentiment_score >= 0.5
                else "SELL"
                if technical_model.directional_bias == "BEARISH" and sentiment_model.sentiment_score < 0.5
                else "HOLD"
            )
            if sizing is None:
                args = {"symbol": symbol, "action": action, "capital": capital}
                sizing = self.mcp.call_tool("risk", "calculate_size", **args)
                self._record_tool_interaction(request_id, "risk", "calculate_size", args, sizing)
                messages.append({"role": "assistant", "iteration": iteration, "content": f"Calculated position sizing for action {action}."})
                continue
            output = run_risk_agent(
                symbol=symbol,
                capital=capital,
                sentiment_output=sentiment_model,
                technical_output=technical_model,
                technical_features=self._active_state_context.get("technical_features") or {},
                request_intent=request_intent,
            )
            messages.append({"role": "assistant", "iteration": iteration, "content": "Risk synthesis completed."})
            return output, messages, errors
        errors.append(f"Risk agent exceeded the {self.max_agent_iterations}-iteration cap.")
        messages.append({"role": "system", "content": errors[-1]})
        technical_model = TechnicalOutput.model_validate(self._active_state_context.get("technical_output"))
        return self._safe_risk_output(request_intent, technical_model), messages, errors

    def _run_coordinator_loop(
        self,
        *,
        request_id: str,
        symbol: str,
        request_intent: RequestIntent,
        scratchpad: list[dict],
    ) -> tuple[CoordinatorOutput, list[dict], list[str]]:
        messages = list(scratchpad)
        errors: list[str] = []
        aggregated = None
        for iteration in range(1, self.max_agent_iterations + 1):
            if aggregated is None:
                aggregated = self.mcp.call_tool("coordinator", "get_all_outputs")
                self._record_tool_interaction(request_id, "coordinator", "get_all_outputs", {}, aggregated)
                messages.append({"role": "tool", "tool": "get_all_outputs", "iteration": iteration, "content": "Loaded all upstream outputs."})
                continue
            output = run_coordinator_agent(
                symbol=symbol,
                request_intent=request_intent,
                sentiment_output=SentimentOutput.model_validate(aggregated["sentiment"]),
                technical_output=TechnicalOutput.model_validate(aggregated["technical"]),
                risk_output=RiskOutput.model_validate(aggregated["risk"]),
            )
            messages.append({"role": "assistant", "iteration": iteration, "content": "Coordinator synthesis completed."})
            return output, messages, errors
        errors.append(f"Coordinator agent exceeded the {self.max_agent_iterations}-iteration cap.")
        messages.append({"role": "system", "content": errors[-1]})
        return self._safe_coordinator_output(symbol, request_intent), messages, errors

    def _chunk_matches_universe_symbol(self, chunk: NewsChunk, stock: StockInfo) -> bool:
        symbol = stock.symbol.upper()
        ticker = str(chunk.metadata.get("ticker", "")).upper()
        tags = {str(tag).upper() for tag in chunk.metadata.get("tags", [])}
        haystack = f"{chunk.text} {chunk.source} {chunk.metadata.get('issuer_name', '')} {chunk.metadata.get('title', '')}".upper()
        normalized_name = stock.name.upper()
        return bool(
            ticker == symbol
            or symbol in tags
            or symbol in haystack
            or normalized_name in haystack
        )

    def _universe_notice_count(self, chunks: list[NewsChunk]) -> int:
        notice_doc_types = {
            "corporate_notices",
            "weekly_notices",
            "monthly_notices",
            "quarterly_notices",
            "issuer_publication",
        }
        return sum(1 for chunk in chunks if chunk.metadata.get("doc_type") in notice_doc_types)

    def _freshness_score(self, chunks: list[NewsChunk]) -> float:
        now = datetime.now(timezone.utc)
        score = 0.0
        for chunk in chunks:
            if chunk.published_at is None:
                continue
            age_days = max(0.0, (now - chunk.published_at).total_seconds() / 86400)
            if age_days <= 3:
                score += 1.0
            elif age_days <= 7:
                score += 0.6
            elif age_days <= 14:
                score += 0.25
        return score

    def _liquidity_score(self, stock: StockInfo, features, policy) -> tuple[float, str]:
        strong_threshold = policy.min_liquidity_mad * 4
        base_threshold = policy.min_liquidity_mad
        weak_threshold = max(25_000.0, policy.min_liquidity_mad * 0.35)
        if stock.last_volume >= strong_threshold and features.zero_volume_bar_count == 0:
            return 0.25, "Liquidite solide"
        if stock.last_volume >= base_threshold and features.zero_volume_bar_count <= 1:
            return 0.15, "Liquidite correcte"
        if stock.last_volume >= weak_threshold:
            return 0.05, "Liquidite limitee"
        return -0.2, "Liquidite insuffisante"

    def _technical_strength_score(self, features, policy) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if features.directional_bias == "BULLISH":
            score += 0.35 * policy.technical_weight
            reasons.append("Biais technique haussier")
        elif features.directional_bias == "NEUTRAL":
            score += 0.1 * policy.technical_weight
            reasons.append("Biais technique neutre")
        if 48 <= features.rsi14 <= 68:
            score += 0.08 + policy.momentum_bonus
            reasons.append("Momentum exploitable")
        compatibility_threshold = min(policy.volatility_ceiling, 0.35 * max(policy.volatility_penalty, 0.1))
        if features.annualized_volatility <= compatibility_threshold:
            score += 0.12
            reasons.append("Volatilite compatible")
        elif features.annualized_volatility >= policy.volatility_ceiling:
            score -= 0.1
            reasons.append("Volatilite elevee")
        return score, reasons

    def _rank_universe_candidate(self, intent: RequestIntent, stock: StockInfo, policy) -> tuple[RankedCandidate | None, str | None]:
        features = analyze_technical_features(stock)
        metadata = {
            "request_id": intent.request_id,
            "symbol": stock.symbol,
            "query_source": "universe_scan",
            "request_mode": intent.request_mode.value,
            "time_horizon": intent.time_horizon.value,
            "risk_preference": intent.risk_preference.value,
        }
        retrieved_chunks = self.retriever.search_news(
            f"{stock.symbol} {stock.name} Morocco",
            top_k=8,
            filters=None,
            metadata=metadata,
        )
        relevant_chunks = [chunk for chunk in retrieved_chunks if self._chunk_matches_universe_symbol(chunk, stock)]
        notice_count = self._universe_notice_count(relevant_chunks)
        freshness_score = self._freshness_score(relevant_chunks)
        score = 0.0
        reasons: list[str] = []

        technical_score, technical_reasons = self._technical_strength_score(features, policy)
        score += technical_score
        reasons.extend(technical_reasons)

        liquidity_score, liquidity_reason = self._liquidity_score(stock, features, policy)
        score += liquidity_score
        reasons.append(liquidity_reason)

        if relevant_chunks:
            catalyst_score = min(0.3 + policy.freshness_bonus, freshness_score * 0.08 * policy.news_weight)
            score += catalyst_score
            reasons.append(f"{len(relevant_chunks)} catalyseur(s) retrouve(s)")
        if notice_count:
            score += min(0.22 + policy.freshness_bonus, notice_count * 0.08 * policy.notice_weight)
            reasons.append(f"{notice_count} avis/publication(s) emetteur")
        if intent.time_horizon.value in {"SHORT_TERM", "INTRADAY"} and freshness_score > 0:
            score += policy.freshness_bonus
            reasons.append("Catalyseurs recents adaptes a l'horizon demande")
        elif intent.time_horizon.value == "SWING" and freshness_score > 0:
            score += min(policy.freshness_bonus, 0.04)
            reasons.append("Catalyseurs compatibles avec une lecture plus large")

        if stock.last_volume < policy.min_liquidity_mad or features.zero_volume_bar_count >= 3:
            return None, f"{stock.symbol}: liquidite trop faible pour un scan exploitable."
        if score < 0.38:
            return None, f"{stock.symbol}: score preliminaire insuffisant ({score:.2f})."
        if not relevant_chunks and policy.horizon_label in {"intraday", "short-term"}:
            return None, f"{stock.symbol}: aucun catalyseur recent pour l'horizon demande."
        if not relevant_chunks and features.directional_bias != "BULLISH":
            return None, f"{stock.symbol}: aucun catalyseur recent et biais technique non haussier."

        return RankedCandidate(symbol=stock.symbol, score=round(score, 4), reasons=reasons), None

    def _run_universe_scan(self, intent: RequestIntent) -> TradeOpportunityList:
        asyncio.run(self.ingest_news())
        stocks = asyncio.run(self.drahmi_client.list_stocks())
        policy = self.policy_engine.build(intent)
        ranked: list[RankedCandidate] = []
        rejected: list[str] = []
        self._active_state_context = {
            "request_id": intent.request_id,
            "request_intent": intent,
        }
        for stock in stocks:
            candidate, rejection_reason = self._rank_universe_candidate(intent, stock, policy)
            if candidate is not None:
                ranked.append(candidate)
            elif rejection_reason is not None:
                rejected.append(rejection_reason)
        ranked.sort(key=lambda item: item.score, reverse=True)
        selected = ranked[:5]
        if not selected:
            return TradeOpportunityList(
                request_id=intent.request_id,
                capital_mad=intent.capital_mad,
                time_horizon=intent.time_horizon,
                risk_preference=intent.risk_preference,
                top_opportunities=[],
                rejected_candidates_summary=rejected or ["Aucune configuration exploitable cette semaine."],
            )
        opportunities: list[TradeOpportunity] = []
        for rank, candidate in enumerate(selected, start=1):
            result = self._run_symbol_legacy(intent, candidate.symbol)
            if result["final_signal"] is not None and result["final_signal"].action != "HOLD":
                opportunities.append(
                    TradeOpportunity(
                        rank=rank,
                        signal=result["final_signal"],
                        coordinator_output=result["coordinator_output"],
                        intent_alignment=result["coordinator_output"].intent_alignment,
                    )
                )
            else:
                rejected.append(
                    f"{candidate.symbol}: aucune configuration exploitable retenue apres evaluation detaillee."
                )
        opportunities = opportunities[:3]
        return TradeOpportunityList(
            request_id=intent.request_id,
            capital_mad=intent.capital_mad,
            time_horizon=intent.time_horizon,
            risk_preference=intent.risk_preference,
            top_opportunities=opportunities,
            rejected_candidates_summary=rejected,
        )

    def _run_symbol_legacy(self, intent: RequestIntent, symbol: str) -> dict[str, Any]:
        asyncio.run(self.ingest_news(symbol))
        self.storage.add_event(intent.request_id, "agent_start", {"agent": "sentiment", "symbol": symbol})
        stock = asyncio.run(self.drahmi_client.get_stock(symbol))
        sentiment_output, sentiment_messages, sentiment_errors = self._run_sentiment_loop(
            request_id=intent.request_id,
            symbol=symbol,
            request_intent=intent,
            scratchpad=[],
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "sentiment"})

        self.storage.add_event(intent.request_id, "agent_start", {"agent": "technical", "symbol": symbol})
        technical_output, technical_features, technical_retry_count, technical_messages, technical_errors = self._run_technical_loop(
            request_id=intent.request_id,
            symbol=symbol,
            stock=stock,
            scratchpad=[],
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "technical"})

        self._active_state_context = {
            "request_id": intent.request_id,
            "symbol": symbol,
            "request_intent": intent,
            "technical_features": technical_features,
            "sentiment_output": sentiment_output,
            "technical_output": technical_output,
        }

        self.storage.add_event(intent.request_id, "agent_start", {"agent": "risk", "symbol": symbol})
        risk_output, risk_messages, risk_errors = self._run_risk_loop(
            request_id=intent.request_id,
            symbol=symbol,
            capital=intent.capital_mad,
            request_intent=intent,
            scratchpad=[],
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "risk"})
        self._active_state_context["risk_output"] = risk_output
        daily_limit = 0.06 if technical_features["is_fixing_mode"] else 0.10
        bias_delta = abs(
            sentiment_output.sentiment_score
            - (
                0.8
                if technical_output.directional_bias == "BULLISH"
                else 0.2
                if technical_output.directional_bias == "BEARISH"
                else 0.5
            )
        )
        human_review_required = (
            risk_output.volatility_estimate > 0.50
            or risk_output.stop_loss_pct > daily_limit * 0.8
            or bias_delta > 0.4
        )

        self.storage.add_event(intent.request_id, "agent_start", {"agent": "coordinator", "symbol": symbol})
        coordinator_output, coordinator_messages, coordinator_errors = self._run_coordinator_loop(
            request_id=intent.request_id,
            symbol=symbol,
            request_intent=intent,
            scratchpad=[],
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "coordinator"})

        final_signal = None
        if not human_review_required:
            final_signal = enforce_limits(
                symbol=symbol,
                request_id=intent.request_id,
                coordinator_output=coordinator_output,
                is_fixing_mode=technical_features["is_fixing_mode"],
                capital=intent.capital_mad,
            )
        return {
            "symbol": symbol,
            "capital": intent.capital_mad,
            "sentiment_output": sentiment_output,
            "technical_output": technical_output,
            "risk_output": risk_output,
            "coordinator_output": coordinator_output,
            "final_signal": final_signal,
            "technical_features": technical_features,
            "technical_retry_count": technical_retry_count,
            "sentiment_messages": sentiment_messages,
            "technical_messages": technical_messages,
            "risk_messages": risk_messages,
            "coordinator_messages": coordinator_messages,
            "human_review_required": human_review_required,
            "errors": [*sentiment_errors, *technical_errors, *risk_errors, *coordinator_errors],
        }

    def _graph_config(self, request_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": request_id}}

    def _initial_graph_state(self, intent: RequestIntent, symbol: str) -> GraphState:
        return {
            "symbol": symbol,
            "capital": intent.capital_mad,
            "request_id": intent.request_id,
            "request_intent": intent.model_dump(mode="json"),
            "candidate_symbols": [],
            "selected_symbol": symbol,
            "intent_warnings": [],
            "stock_info": None,
            "news_chunks": [],
            "technical_features": None,
            "sentiment_output": None,
            "technical_output": None,
            "risk_output": None,
            "coordinator_output": None,
            "final_signal": None,
            "sentiment_messages": [],
            "technical_messages": [],
            "risk_messages": [],
            "coordinator_messages": [],
            "technical_retry_count": 0,
            "human_review_required": False,
            "human_review_decision": None,
            "started_at": datetime.now(timezone.utc),
            "errors": [],
        }

    def _prepare_context_node(self, state: GraphState) -> dict[str, Any]:
        symbol = state["symbol"]
        request_intent = RequestIntent.model_validate(state["request_intent"])
        self._active_state_context = {
            "request_id": state["request_id"],
            "symbol": symbol,
            "request_intent": request_intent,
        }
        asyncio.run(self.ingest_news(symbol))
        stock = asyncio.run(self.drahmi_client.get_stock(symbol))
        news_chunks = self.retriever.search_news(
            f"{symbol} Morocco",
            top_k=5,
            filters=None,
            metadata=self._retrieval_metadata(query_source="prepare_context", top_k=5),
        )
        return {
            "selected_symbol": symbol,
            "stock_info": stock.model_dump(mode="json"),
            "news_chunks": [chunk.model_dump(mode="json") for chunk in news_chunks],
        }

    def _sentiment_node(self, state: GraphState) -> dict[str, Any]:
        request_intent = RequestIntent.model_validate(state["request_intent"])
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "sentiment", "symbol": state["symbol"]})
        sentiment_output, sentiment_messages, sentiment_errors = self._run_sentiment_loop(
            request_id=state["request_id"],
            symbol=state["symbol"],
            request_intent=request_intent,
            scratchpad=state.get("sentiment_messages", []),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "sentiment"})
        return {
            "sentiment_output": sentiment_output.model_dump(mode="json"),
            "sentiment_messages": sentiment_messages,
            "errors": sentiment_errors,
        }

    def _technical_node(self, state: GraphState) -> dict[str, Any]:
        stock = StockInfo.model_validate(state["stock_info"])
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "technical", "symbol": state["symbol"]})
        technical_output, technical_features, technical_retry_count, technical_messages, technical_errors = self._run_technical_loop(
            request_id=state["request_id"],
            symbol=state["symbol"],
            stock=stock,
            scratchpad=state.get("technical_messages", []),
            initial_retry_count=state.get("technical_retry_count", 0),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "technical"})
        return {
            "technical_output": technical_output.model_dump(mode="json"),
            "technical_features": technical_features,
            "technical_retry_count": technical_retry_count,
            "technical_messages": technical_messages,
            "errors": technical_errors,
        }

    def _risk_node(self, state: GraphState) -> dict[str, Any]:
        request_intent = RequestIntent.model_validate(state["request_intent"])
        sentiment_output = SentimentOutput.model_validate(state["sentiment_output"])
        technical_output = TechnicalOutput.model_validate(state["technical_output"])
        technical_features = state["technical_features"] or {}
        self._active_state_context = {
            "request_id": state["request_id"],
            "symbol": state["symbol"],
            "request_intent": request_intent,
            "technical_features": technical_features,
            "sentiment_output": sentiment_output,
            "technical_output": technical_output,
        }
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "risk", "symbol": state["symbol"]})
        risk_output, risk_messages, risk_errors = self._run_risk_loop(
            request_id=state["request_id"],
            symbol=state["symbol"],
            capital=state["capital"],
            request_intent=request_intent,
            scratchpad=state.get("risk_messages", []),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "risk"})
        self._active_state_context["risk_output"] = risk_output
        daily_limit = 0.06 if technical_features.get("is_fixing_mode") else 0.10
        bias_delta = abs(
            sentiment_output.sentiment_score
            - (
                0.8
                if technical_output.directional_bias == "BULLISH"
                else 0.2
                if technical_output.directional_bias == "BEARISH"
                else 0.5
            )
        )
        human_review_required = (
            risk_output.volatility_estimate > 0.50
            or risk_output.stop_loss_pct > daily_limit * 0.8
            or bias_delta > 0.4
        )
        return {
            "risk_output": risk_output.model_dump(mode="json"),
            "risk_messages": risk_messages,
            "human_review_required": human_review_required,
            "errors": risk_errors,
        }

    def _risk_router(self, state: GraphState) -> str:
        return "human_review" if state.get("human_review_required") else "coordinator_agent"

    def _human_review_node(self, state: GraphState) -> dict[str, Any]:
        decision = interrupt(
            {
                "request_id": state["request_id"],
                "symbol": state["symbol"],
                "reason": "Human review required before coordinator execution.",
                "risk_output": state.get("risk_output"),
            }
        )
        return {"human_review_decision": decision}

    def _human_review_router(self, state: GraphState) -> str:
        return "coordinator_agent" if state.get("human_review_decision") == "approved" else "end"

    def _coordinator_node(self, state: GraphState) -> dict[str, Any]:
        request_intent = RequestIntent.model_validate(state["request_intent"])
        self._active_state_context = {
            "request_id": state["request_id"],
            "symbol": state["symbol"],
            "request_intent": request_intent,
            "technical_features": state.get("technical_features") or {},
            "sentiment_output": SentimentOutput.model_validate(state["sentiment_output"]),
            "technical_output": TechnicalOutput.model_validate(state["technical_output"]),
            "risk_output": RiskOutput.model_validate(state["risk_output"]),
        }
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "coordinator", "symbol": state["symbol"]})
        coordinator_output, coordinator_messages, coordinator_errors = self._run_coordinator_loop(
            request_id=state["request_id"],
            symbol=state["symbol"],
            request_intent=request_intent,
            scratchpad=state.get("coordinator_messages", []),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "coordinator"})
        return {
            "coordinator_output": coordinator_output.model_dump(mode="json"),
            "coordinator_messages": coordinator_messages,
            "errors": coordinator_errors,
        }

    def _enforce_limits_node(self, state: GraphState) -> dict[str, Any]:
        technical_features = state["technical_features"] or {}
        final_signal = enforce_limits(
            symbol=state["symbol"],
            request_id=state["request_id"],
            coordinator_output=CoordinatorOutput.model_validate(state["coordinator_output"]),
            is_fixing_mode=technical_features.get("is_fixing_mode", False),
            capital=state["capital"],
        )
        return {"final_signal": final_signal.model_dump(mode="json")}

    def _coerce_model(self, value: Any, model_cls):
        if value is None:
            return None
        if isinstance(value, model_cls):
            return value
        return model_cls.model_validate(value)
