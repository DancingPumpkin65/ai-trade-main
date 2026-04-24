from __future__ import annotations

import asyncio
from datetime import datetime

from trading_agents.core.broker.alpaca import AlpacaPreviewService
from trading_agents.core.data.drahmi import DrahmiClient
from trading_agents.core.data.news_global import MarketAuxClient
from trading_agents.core.data.news_morocco import MoroccoNewsClient
from trading_agents.core.intent.policy import IntentPolicyEngine
from trading_agents.core.mcp.server import MCPServer
from trading_agents.core.models import (
    CoordinatorOutput,
    GenerateSignalRequest,
    RankedCandidate,
    RequestIntent,
    RequestMode,
    SignalStatus,
    TradeOpportunity,
    TradeOpportunityList,
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
from trading_agents.graph.technical_node import run_technical_agent


class TradingGraphService:
    def __init__(
        self,
        *,
        storage: Storage,
        drahmi_client: DrahmiClient,
        morocco_news_client: MoroccoNewsClient,
        marketaux_client: MarketAuxClient,
        alpaca_preview_service: AlpacaPreviewService,
    ):
        self.storage = storage
        self.drahmi_client = drahmi_client
        self.morocco_news_client = morocco_news_client
        self.marketaux_client = marketaux_client
        self.alpaca_preview_service = alpaca_preview_service
        self.policy_engine = IntentPolicyEngine()
        self.vector_store = InMemoryVectorStore()
        self.indexer = Indexer(self.vector_store)
        self.retriever = NewsRetriever(self.vector_store)
        self.mcp = MCPServer()
        self._register_tools()

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
        result = self._run_symbol(intent, intent.symbols_requested[0], persist_intermediate=True)
        if result["human_review_required"]:
            self.storage.update_request(
                intent.request_id,
                status=SignalStatus.WAITING_HUMAN,
                human_review_required=True,
                coordinator_output=result.get("coordinator_output"),
                errors=result["errors"],
                state=result,
            )
            self.storage.add_event(intent.request_id, "human_review_required", {"symbol": intent.symbols_requested[0]})
            return

        final_signal = result["final_signal"]
        coordinator_output = result["coordinator_output"]
        self.storage.update_request(
            intent.request_id,
            status=SignalStatus.COMPLETED,
            human_review_required=False,
            final_signal=final_signal,
            coordinator_output=coordinator_output,
            errors=result["errors"],
        )
        self.storage.add_event(intent.request_id, "pipeline_complete", {"symbol": final_signal.symbol})

    def resume(self, request_id: str, decision: str) -> None:
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
            result = self._run_symbol(intent, candidate.symbol, persist_intermediate=False)
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

    def _run_symbol(self, intent: RequestIntent, symbol: str, *, persist_intermediate: bool) -> dict:
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
        technical_output, technical_features = run_technical_agent(stock)
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
        bias_delta = abs(sentiment_output.sentiment_score - (0.8 if technical_output.directional_bias == "BULLISH" else 0.2 if technical_output.directional_bias == "BEARISH" else 0.5))
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
            "human_review_required": human_review_required,
            "errors": [],
        }
