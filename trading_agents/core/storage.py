from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from trading_agents.core.models import (
    AlpacaOrderIntent,
    AlpacaOrderStatus,
    RequestIntent,
    SignalRecord,
    SignalStatus,
    TradeOpportunityList,
    TradingSignal,
    CoordinatorOutput,
)


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_requests (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    raw_prompt TEXT,
                    request_intent_json TEXT NOT NULL,
                    parser_confidence REAL NOT NULL,
                    extraction_method TEXT NOT NULL,
                    human_review_required INTEGER NOT NULL DEFAULT 0,
                    final_signal_json TEXT,
                    opportunity_list_json TEXT,
                    coordinator_output_json TEXT,
                    alpaca_order_json TEXT,
                    alpaca_order_status TEXT NOT NULL DEFAULT 'NOT_PREPARED',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    state_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_request(self, request_id: str, intent: RequestIntent) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(intent.model_dump(mode="json"), default=_json_default)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO signal_requests (
                    request_id, status, raw_prompt, request_intent_json, parser_confidence,
                    extraction_method, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    SignalStatus.PENDING.value,
                    intent.raw_prompt,
                    payload,
                    intent.parser_confidence,
                    intent.extraction_method,
                    now,
                    now,
                ),
            )

    def update_request(
        self,
        request_id: str,
        *,
        status: SignalStatus | None = None,
        human_review_required: bool | None = None,
        final_signal: TradingSignal | None = None,
        opportunity_list: TradeOpportunityList | None = None,
        coordinator_output: CoordinatorOutput | None = None,
        alpaca_order: AlpacaOrderIntent | None = None,
        alpaca_order_status: AlpacaOrderStatus | None = None,
        errors: list[str] | None = None,
        state: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        fields: list[str] = ["updated_at = ?"]
        values: list = [now]
        if status is not None:
            fields.append("status = ?")
            values.append(status.value)
        if human_review_required is not None:
            fields.append("human_review_required = ?")
            values.append(1 if human_review_required else 0)
        if final_signal is not None:
            fields.append("final_signal_json = ?")
            values.append(json.dumps(final_signal.model_dump(mode="json"), default=_json_default))
        if opportunity_list is not None:
            fields.append("opportunity_list_json = ?")
            values.append(json.dumps(opportunity_list.model_dump(mode="json"), default=_json_default))
        if coordinator_output is not None:
            fields.append("coordinator_output_json = ?")
            values.append(json.dumps(coordinator_output.model_dump(mode="json"), default=_json_default))
        if alpaca_order is not None:
            fields.append("alpaca_order_json = ?")
            values.append(json.dumps(alpaca_order.model_dump(mode="json"), default=_json_default))
        if alpaca_order_status is not None:
            fields.append("alpaca_order_status = ?")
            values.append(alpaca_order_status.value)
        if errors is not None:
            fields.append("errors_json = ?")
            values.append(json.dumps(errors))
        if state is not None:
            fields.append("state_json = ?")
            values.append(json.dumps(state, default=_json_default))
        values.append(request_id)

        with self.connection() as conn:
            conn.execute(f"UPDATE signal_requests SET {', '.join(fields)} WHERE request_id = ?", values)

    def add_audit_log(self, request_id: str | None, event_type: str, message: str, payload: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (request_id, event_type, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    event_type,
                    message,
                    json.dumps(payload, default=_json_default),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def add_event(self, request_id: str, event_type: str, payload: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO signal_events (request_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    request_id,
                    event_type,
                    json.dumps(payload, default=_json_default),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_events(self, request_id: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT event_type, payload_json, created_at FROM signal_events WHERE request_id = ? ORDER BY id ASC",
                (request_id,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_request_row(self, request_id: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM signal_requests WHERE request_id = ?", (request_id,)).fetchone()

    def get_signal_record(self, request_id: str) -> SignalRecord | None:
        row = self.get_request_row(request_id)
        if row is None:
            return None
        intent = RequestIntent.model_validate_json(row["request_intent_json"])
        final_signal = TradingSignal.model_validate_json(row["final_signal_json"]) if row["final_signal_json"] else None
        opportunity_list = (
            TradeOpportunityList.model_validate_json(row["opportunity_list_json"])
            if row["opportunity_list_json"]
            else None
        )
        coordinator_output = (
            CoordinatorOutput.model_validate_json(row["coordinator_output_json"])
            if row["coordinator_output_json"]
            else None
        )
        alpaca_order = (
            AlpacaOrderIntent.model_validate_json(row["alpaca_order_json"]) if row["alpaca_order_json"] else None
        )
        return SignalRecord(
            request_id=row["request_id"],
            status=SignalStatus(row["status"]),
            request_intent=intent,
            final_signal=final_signal,
            opportunity_list=opportunity_list,
            coordinator_output=coordinator_output,
            human_review_required=bool(row["human_review_required"]),
            alpaca_order_status=AlpacaOrderStatus(row["alpaca_order_status"]),
            alpaca_order=alpaca_order,
            errors=json.loads(row["errors_json"]),
        )

    def list_history(self, limit: int = 50) -> list[SignalRecord]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT request_id FROM signal_requests ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        records: list[SignalRecord] = []
        for row in rows:
            record = self.get_signal_record(row["request_id"])
            if record:
                records.append(record)
        return records

    def get_saved_state(self, request_id: str) -> dict | None:
        row = self.get_request_row(request_id)
        if row is None or not row["state_json"]:
            return None
        return json.loads(row["state_json"])
