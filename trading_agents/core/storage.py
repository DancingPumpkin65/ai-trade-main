from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_agents.core.database import DatabaseAdapter, build_database_adapter
from trading_agents.core.models import (
    AlpacaOrderIntent,
    AlpacaOrderStatus,
    RequestIntent,
    SignalRecord,
    SignalStatus,
    TradeOpportunityList,
    TradingSignal,
    CoordinatorOutput,
    UniverseScanCandidateRecord,
)
from trading_agents.core.migrations import MigrationRunner


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


class Storage:
    def __init__(self, db_path: Path, database_url: str | None = None):
        self.db_path = db_path
        self.database_url = database_url
        self.adapter: DatabaseAdapter = build_database_adapter(database_url, db_path)
        self.adapter.ensure_parent_dirs()
        self.migration_runner = MigrationRunner(db_path, database_url)
        self.init_db()

    def connection(self):
        return self.adapter.connection()

    def init_db(self) -> None:
        self.migration_runner.migrate()

    @property
    def schema_version(self) -> int:
        return self.migration_runner.current_version()

    @property
    def backend_name(self) -> str:
        return self.adapter.backend_name

    def create_request(self, request_id: str, intent: RequestIntent) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(intent.model_dump(mode="json"), default=_json_default)
        with self.connection() as conn:
            conn.execute(
                self.adapter.prepare_sql(
                    """
                INSERT INTO signal_requests (
                    request_id, status, raw_prompt, request_intent_json, parser_confidence,
                    extraction_method, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                ),
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

    def user_exists(self, username: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(self.adapter.prepare_sql("SELECT 1 FROM users WHERE username = ?"), (username,)).fetchone()
        return row is not None

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
            conn.execute(
                self.adapter.prepare_sql(f"UPDATE signal_requests SET {', '.join(fields)} WHERE request_id = ?"),
                values,
            )

    def add_audit_log(self, request_id: str | None, event_type: str, message: str, payload: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                self.adapter.prepare_sql(
                    """
                INSERT INTO audit_log (request_id, event_type, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """
                ),
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
                self.adapter.prepare_sql(
                    """
                INSERT INTO signal_events (request_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """
                ),
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
                self.adapter.prepare_sql(
                    "SELECT event_type, payload_json, created_at FROM signal_events WHERE request_id = ? ORDER BY id ASC"
                ),
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

    def get_events_after(self, request_id: str, after_id: int = 0) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                self.adapter.prepare_sql(
                    """
                SELECT id, event_type, payload_json, created_at
                FROM signal_events
                WHERE request_id = ? AND id > ?
                ORDER BY id ASC
                """
                ),
                (request_id, after_id),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_request_row(self, request_id: str):
        with self.connection() as conn:
            return conn.execute(
                self.adapter.prepare_sql("SELECT * FROM signal_requests WHERE request_id = ?"),
                (request_id,),
            ).fetchone()

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
                self.adapter.prepare_sql("SELECT request_id FROM signal_requests ORDER BY updated_at DESC LIMIT ?"),
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

    def upsert_opportunity_alpaca_order(
        self,
        request_id: str,
        symbol: str,
        order: AlpacaOrderIntent,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                self.adapter.prepare_sql(
                    """
                INSERT INTO opportunity_alpaca_orders (
                    request_id, symbol, alpaca_order_json, alpaca_order_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, symbol) DO UPDATE SET
                    alpaca_order_json = excluded.alpaca_order_json,
                    alpaca_order_status = excluded.alpaca_order_status,
                    updated_at = excluded.updated_at
                """
                ),
                (
                    request_id,
                    symbol.upper(),
                    json.dumps(order.model_dump(mode="json"), default=_json_default),
                    order.status.value,
                    timestamp,
                    timestamp,
                ),
            )

    def get_opportunity_alpaca_order(self, request_id: str, symbol: str) -> AlpacaOrderIntent | None:
        with self.connection() as conn:
            row = conn.execute(
                self.adapter.prepare_sql(
                    """
                SELECT alpaca_order_json
                FROM opportunity_alpaca_orders
                WHERE request_id = ? AND symbol = ?
                """
                ),
                (request_id, symbol.upper()),
            ).fetchone()
        if row is None:
            return None
        return AlpacaOrderIntent.model_validate_json(row["alpaca_order_json"])

    def list_opportunity_alpaca_orders(self, request_id: str) -> dict[str, AlpacaOrderIntent]:
        with self.connection() as conn:
            rows = conn.execute(
                self.adapter.prepare_sql(
                    """
                SELECT symbol, alpaca_order_json
                FROM opportunity_alpaca_orders
                WHERE request_id = ?
                ORDER BY symbol ASC
                """
                ),
                (request_id,),
            ).fetchall()
        return {
            row["symbol"]: AlpacaOrderIntent.model_validate_json(row["alpaca_order_json"])
            for row in rows
        }

    def replace_universe_scan_candidates(self, request_id: str, candidates: list[UniverseScanCandidateRecord]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                self.adapter.prepare_sql("DELETE FROM universe_scan_candidates WHERE request_id = ?"),
                (request_id,),
            )
            conn.executemany(
                self.adapter.prepare_sql(
                    """
                INSERT INTO universe_scan_candidates (
                    request_id, symbol, score, reasons_json, selected_for_deep_eval,
                    rank_position, evaluation_status, rejection_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                ),
                [
                    (
                        candidate.request_id,
                        candidate.symbol,
                        candidate.score,
                        json.dumps(candidate.reasons),
                        1 if candidate.selected_for_deep_eval else 0,
                        candidate.rank_position,
                        candidate.evaluation_status,
                        candidate.rejection_reason,
                        timestamp,
                    )
                    for candidate in candidates
                ],
            )

    def get_universe_scan_candidates(self, request_id: str) -> list[UniverseScanCandidateRecord]:
        with self.connection() as conn:
            rows = conn.execute(
                self.adapter.prepare_sql(
                    """
                SELECT request_id, symbol, score, reasons_json, selected_for_deep_eval,
                       rank_position, evaluation_status, rejection_reason
                FROM universe_scan_candidates
                WHERE request_id = ?
                ORDER BY
                    CASE WHEN rank_position IS NULL THEN 1 ELSE 0 END,
                    rank_position ASC,
                    score DESC,
                    symbol ASC
                """
                ),
                (request_id,),
            ).fetchall()
        return [
            UniverseScanCandidateRecord(
                request_id=row["request_id"],
                symbol=row["symbol"],
                score=row["score"],
                reasons=json.loads(row["reasons_json"]),
                selected_for_deep_eval=bool(row["selected_for_deep_eval"]),
                rank_position=row["rank_position"],
                evaluation_status=row["evaluation_status"],
                rejection_reason=row["rejection_reason"],
            )
            for row in rows
        ]
