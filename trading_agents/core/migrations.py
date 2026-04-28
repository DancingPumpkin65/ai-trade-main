from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from trading_agents.core.config import get_settings


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
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
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS universe_scan_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                score REAL,
                reasons_json TEXT NOT NULL DEFAULT '[]',
                selected_for_deep_eval INTEGER NOT NULL DEFAULT 0,
                rank_position INTEGER,
                evaluation_status TEXT NOT NULL,
                rejection_reason TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS opportunity_alpaca_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                alpaca_order_json TEXT NOT NULL,
                alpaca_order_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(request_id, symbol)
            )
            """,
        ),
    ),
    Migration(
        version=2,
        name="request_indexes",
        statements=(
            "CREATE INDEX IF NOT EXISTS idx_signal_requests_updated_at ON signal_requests(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_signal_events_request_id_id ON signal_events(request_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_log_request_id_created_at ON audit_log(request_id, created_at)",
            """
            CREATE INDEX IF NOT EXISTS idx_universe_scan_candidates_request_rank
            ON universe_scan_candidates(request_id, rank_position, symbol)
            """,
        ),
    ),
)


class MigrationRunner:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_migrations_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )

    def applied_versions(self) -> set[int]:
        with self.connection() as conn:
            self._ensure_migrations_table(conn)
            rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {int(row["version"]) for row in rows}

    def current_version(self) -> int:
        with self.connection() as conn:
            self._ensure_migrations_table(conn)
            row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations").fetchone()
        if row is None:
            return 0
        return int(row["version"])

    def migrate(self) -> list[Migration]:
        applied_versions = self.applied_versions()
        applied_now: list[Migration] = []
        with self.connection() as conn:
            self._ensure_migrations_table(conn)
            for migration in MIGRATIONS:
                if migration.version in applied_versions:
                    continue
                for statement in migration.statements:
                    conn.execute(statement)
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (migration.version, migration.name),
                )
                applied_now.append(migration)
        return applied_now


def cli_main() -> None:
    settings = get_settings()
    runner = MigrationRunner(settings.db_path)
    applied = runner.migrate()
    print(
        json.dumps(
            {
                "db_path": str(settings.db_path),
                "current_version": runner.current_version(),
                "applied_migrations": [
                    {"version": migration.version, "name": migration.name}
                    for migration in applied
                ],
            }
        )
    )
