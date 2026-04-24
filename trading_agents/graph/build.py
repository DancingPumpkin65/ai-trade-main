from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agents.core.broker.alpaca import AlpacaPreviewService
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
from trading_agents.core.rag.indexer import Indexer
from trading_agents.core.rag.retriever import NewsRetriever
from trading_agents.core.rag.store import InMemoryVectorStore
from trading_agents.core.storage import Storage
from trading_agents.graph.coordinator_node import run_coordinator_agent
from trading_agents.graph.enforce_limits import enforce_limits
from trading_agents.graph.helpers import analyze_technical_features
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
    ):
        self.storage = storage
        self.drahmi_client = drahmi_client
        self.morocco_news_client = morocco_news_client
        self.marketaux_client = marketaux_client
        self.alpaca_preview_service = alpaca_preview_service
        self.checkpoint_path = checkpoint_path
        self.policy_engine = IntentPolicyEngine()
        self.vector_store = InMemoryVectorStore()
        self.indexer = Indexer(self.vector_store)
        self.retriever = NewsRetriever(self.vector_store)
        self.mcp = MCPServer()
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
        self.mcp.register_tool("sentiment", "search_news", lambda query, top_k=5, filters=None: self.retriever.search_news(query, top_k, filters))
        self.mcp.register_tool("sentiment", "get_market_data", lambda symbol: asyncio.run(self.drahmi_client.get_stock(symbol)))
        self.mcp.register_tool("technical", "get_market_data", lambda symbol: asyncio.run(self.drahmi_client.get_stock(symbol)))
        self.mcp.register_tool("technical", "analyze_technical", lambda symbol: analyze_technical_features(asyncio.run(self.drahmi_client.get_stock(symbol))))

    async def ingest_news(self, symbol: str | None = None) -> None:
        morocco_news = await self.morocco_news_client.fetch()
        self.indexer.upsert_news(morocco_news)
        if symbol:
            global_news = await self.marketaux_client.fetch_for_symbol(symbol)
            self.indexer.upsert_news(global_news)

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

    def _run_universe_scan(self, intent: RequestIntent) -> TradeOpportunityList:
        asyncio.run(self.ingest_news())
        stocks = asyncio.run(self.drahmi_client.list_stocks())
        policy = self.policy_engine.build(intent)
        ranked: list[RankedCandidate] = []
        for stock in stocks:
            features = analyze_technical_features(stock)
            score = 0.0
            reasons = []
            if features.directional_bias == "BULLISH":
                score += 0.5 * policy.technical_weight
                reasons.append("Directional bias bullish")
            if stock.last_volume > 0:
                score += 0.2
                reasons.append("Liquidity available")
            score += max(0.0, 0.3 - features.annualized_volatility * 0.2 / max(policy.volatility_penalty, 0.1))
            if intent.time_horizon.value == "SHORT_TERM":
                score += 0.1
            ranked.append(RankedCandidate(symbol=stock.symbol, score=round(score, 4), reasons=reasons))
        ranked.sort(key=lambda item: item.score, reverse=True)
        selected = ranked[:5]
        opportunities: list[TradeOpportunity] = []
        rejected: list[str] = []
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
                rejected.append(f"{candidate.symbol}: aucune configuration exploitable retenue.")
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
        news_chunks = self.retriever.search_news(f"{symbol} Morocco", top_k=5, filters=None)
        sentiment_output = run_sentiment_agent(
            symbol=symbol,
            request_intent=intent,
            market_data=stock,
            news_chunks=news_chunks,
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "sentiment"})

        self.storage.add_event(intent.request_id, "agent_start", {"agent": "technical", "symbol": symbol})
        technical_output, technical_features, technical_retry_count, technical_errors = self._execute_technical_with_ground_truth(
            request_id=intent.request_id,
            symbol=symbol,
            stock=stock,
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "technical"})

        self.storage.add_event(intent.request_id, "agent_start", {"agent": "risk", "symbol": symbol})
        risk_output = run_risk_agent(
            symbol=symbol,
            capital=intent.capital_mad,
            sentiment_output=sentiment_output,
            technical_output=technical_output,
            technical_features=technical_features,
            request_intent=intent,
        )
        self.storage.add_event(intent.request_id, "agent_complete", {"agent": "risk"})
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
        coordinator_output = run_coordinator_agent(
            symbol=symbol,
            request_intent=intent,
            sentiment_output=sentiment_output,
            technical_output=technical_output,
            risk_output=risk_output,
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
            "human_review_required": human_review_required,
            "errors": technical_errors,
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
        asyncio.run(self.ingest_news(symbol))
        stock = asyncio.run(self.drahmi_client.get_stock(symbol))
        news_chunks = self.retriever.search_news(f"{symbol} Morocco", top_k=5, filters=None)
        return {
            "selected_symbol": symbol,
            "stock_info": stock.model_dump(mode="json"),
            "news_chunks": [chunk.model_dump(mode="json") for chunk in news_chunks],
        }

    def _sentiment_node(self, state: GraphState) -> dict[str, Any]:
        request_intent = RequestIntent.model_validate(state["request_intent"])
        stock = StockInfo.model_validate(state["stock_info"])
        news_chunks = [NewsChunk.model_validate(chunk) for chunk in state.get("news_chunks", [])]
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "sentiment", "symbol": state["symbol"]})
        sentiment_output = run_sentiment_agent(
            symbol=state["symbol"],
            request_intent=request_intent,
            market_data=stock,
            news_chunks=news_chunks,
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "sentiment"})
        return {"sentiment_output": sentiment_output.model_dump(mode="json")}

    def _technical_node(self, state: GraphState) -> dict[str, Any]:
        stock = StockInfo.model_validate(state["stock_info"])
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "technical", "symbol": state["symbol"]})
        technical_output, technical_features, technical_retry_count, technical_errors = self._execute_technical_with_ground_truth(
            request_id=state["request_id"],
            symbol=state["symbol"],
            stock=stock,
            initial_retry_count=state.get("technical_retry_count", 0),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "technical"})
        return {
            "technical_output": technical_output.model_dump(mode="json"),
            "technical_features": technical_features,
            "technical_retry_count": technical_retry_count,
            "errors": [*state.get("errors", []), *technical_errors],
        }

    def _risk_node(self, state: GraphState) -> dict[str, Any]:
        request_intent = RequestIntent.model_validate(state["request_intent"])
        sentiment_output = SentimentOutput.model_validate(state["sentiment_output"])
        technical_output = TechnicalOutput.model_validate(state["technical_output"])
        technical_features = state["technical_features"] or {}
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "risk", "symbol": state["symbol"]})
        risk_output = run_risk_agent(
            symbol=state["symbol"],
            capital=state["capital"],
            sentiment_output=sentiment_output,
            technical_output=technical_output,
            technical_features=technical_features,
            request_intent=request_intent,
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "risk"})
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
            "human_review_required": human_review_required,
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
        self.storage.add_event(state["request_id"], "agent_start", {"agent": "coordinator", "symbol": state["symbol"]})
        coordinator_output = run_coordinator_agent(
            symbol=state["symbol"],
            request_intent=request_intent,
            sentiment_output=SentimentOutput.model_validate(state["sentiment_output"]),
            technical_output=TechnicalOutput.model_validate(state["technical_output"]),
            risk_output=RiskOutput.model_validate(state["risk_output"]),
        )
        self.storage.add_event(state["request_id"], "agent_complete", {"agent": "coordinator"})
        return {"coordinator_output": coordinator_output.model_dump(mode="json")}

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
