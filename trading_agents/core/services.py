from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timezone

from trading_agents.core.broker.alpaca import AlpacaPreviewService
from trading_agents.core.config import Settings
from trading_agents.core.data.drahmi import DrahmiClient
from trading_agents.core.data.news_global import MarketAuxClient
from trading_agents.core.data.news_morocco import MoroccoNewsClient
from trading_agents.core.intent.normalizer import OllamaIntentNormalizer
from trading_agents.core.intent.parser import IntentParser
from trading_agents.core.models import GenerateSignalRequest, GenerateSignalResponse, SignalRecord, SignalStatus
from trading_agents.core.observability import LangSmithIntentTracer
from trading_agents.core.storage import Storage
from trading_agents.graph.build import TradingGraphService


class AuthService:
    def __init__(self, storage: Storage, secret_key: str):
        self.storage = storage
        self.secret_key = secret_key.encode("utf-8")

    def register(self, username: str, password: str) -> dict:
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        with self.storage.connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, datetime.now(timezone.utc).isoformat()),
            )
        return {"username": username}

    def login(self, username: str, password: str) -> dict:
        with self.storage.connection() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            raise ValueError("Invalid credentials.")
        expected = row["password_hash"]
        provided = hashlib.sha256(password.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(expected, provided):
            raise ValueError("Invalid credentials.")
        signature = hmac.new(self.secret_key, username.encode("utf-8"), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{username}:{signature}".encode("utf-8")).decode("utf-8")
        return {"access_token": token, "token_type": "bearer"}


class AppServices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage(settings.db_path)
        intent_tracer = LangSmithIntentTracer(
            enabled=settings.langsmith_tracing,
            project_name=settings.langsmith_project,
        )
        self.intent_parser = IntentParser(
            normalizer=OllamaIntentNormalizer(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            ),
            tracer=intent_tracer,
        )
        self.alpaca_preview_service = AlpacaPreviewService(enabled=settings.alpaca_enabled)
        self.drahmi_client = DrahmiClient(settings.drahmi_base_url, settings.drahmi_api_key, settings.drahmi_daily_limit)
        self.morocco_news_client = MoroccoNewsClient()
        self.marketaux_client = MarketAuxClient(settings.marketaux_base_url, settings.marketaux_api_key)
        self.graph_service = TradingGraphService(
            storage=self.storage,
            drahmi_client=self.drahmi_client,
            morocco_news_client=self.morocco_news_client,
            marketaux_client=self.marketaux_client,
            alpaca_preview_service=self.alpaca_preview_service,
            checkpoint_path=settings.langgraph_checkpoint_path,
            chroma_persist_dir=settings.chroma_persist_dir,
            bourse_cache_dir=settings.data_dir / "bourse_pdfs",
            env=settings.env,
            langsmith_tracing=settings.langsmith_tracing,
            langsmith_project=settings.langsmith_project,
        )
        self.auth_service = AuthService(self.storage, settings.secret_key)

    def close(self) -> None:
        self.graph_service.close()

    def health(self) -> dict:
        return {
            "status": "ok",
            "db_path": str(self.settings.db_path),
            "langgraph_checkpoint_path": str(self.settings.langgraph_checkpoint_path),
            "langsmith_tracing": self.settings.langsmith_tracing,
            "alpaca_enabled": self.settings.alpaca_enabled,
            "langgraph_enabled": self.graph_service.langgraph_enabled,
            "rag_backend": self.graph_service.vector_store.backend_name,
            "bourse_cache_dir": str(self.settings.data_dir / "bourse_pdfs"),
        }

    def generate(self, payload: GenerateSignalRequest) -> GenerateSignalResponse:
        intent = self.intent_parser.parse(payload)
        self.storage.create_request(intent.request_id, intent)
        self.storage.add_audit_log(
            intent.request_id,
            "intent_parsed",
            "Request intent normalized.",
            {
                "raw_prompt": intent.raw_prompt,
                "symbols_requested": intent.symbols_requested,
                "request_mode": intent.request_mode.value,
                "risk_preference": intent.risk_preference.value,
                "time_horizon": intent.time_horizon.value,
                "bias_override_refused": intent.bias_override_refused,
                "parser_confidence": intent.parser_confidence,
                "extraction_method": intent.extraction_method,
            },
        )
        self.graph_service.start(intent)
        record = self.storage.get_signal_record(intent.request_id)
        return GenerateSignalResponse(request_id=intent.request_id, status=record.status if record else SignalStatus.FAILED)

    def get_signal(self, request_id: str) -> SignalRecord:
        record = self.storage.get_signal_record(request_id)
        if record is None:
            raise ValueError("Signal request not found.")
        return record

    def approve(self, request_id: str) -> SignalRecord:
        self.storage.add_audit_log(request_id, "human_review", "Operator approved request.", {"decision": "approved"})
        self.graph_service.resume(request_id, "approved")
        return self.get_signal(request_id)

    def reject(self, request_id: str) -> SignalRecord:
        self.storage.add_audit_log(request_id, "human_review", "Operator rejected request.", {"decision": "rejected"})
        self.graph_service.resume(request_id, "rejected")
        return self.get_signal(request_id)

    def history(self) -> list[SignalRecord]:
        return self.storage.list_history()

    def stream_events(self, request_id: str) -> list[dict]:
        return self.storage.get_events(request_id)

    def export_signal_detail(self, request_id: str) -> dict:
        record = self.get_signal(request_id)
        payload = record.model_dump(mode="json")
        payload["signal_status"] = record.status.value
        payload["human_review_required"] = record.human_review_required
        payload["alpaca_order_status"] = record.alpaca_order_status.value
        return payload

    def get_alpaca_order(self, request_id: str) -> dict:
        record = self.get_signal(request_id)
        return {
            "request_id": request_id,
            "alpaca_order_status": record.alpaca_order_status.value,
            "alpaca_order": record.alpaca_order.model_dump(mode="json") if record.alpaca_order else None,
        }
