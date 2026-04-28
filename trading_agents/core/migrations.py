from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trading_agents.core.config import get_settings
from trading_agents.core.database import DatabaseAdapter, build_database_adapter


@dataclass(frozen=True)
class Migration:
    version: int
    name: str

    sqlite_statements: tuple[str, ...]
    postgres_statements: tuple[str, ...] | None = None

    def statements_for_backend(self, backend_name: str) -> tuple[str, ...]:
        if backend_name == "postgresql" and self.postgres_statements is not None:
            return self.postgres_statements
        return self.sqlite_statements


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_schema",
        sqlite_statements=(
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
        postgres_statements=(
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
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
                id BIGSERIAL PRIMARY KEY,
                request_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signal_events (
                id BIGSERIAL PRIMARY KEY,
                request_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS universe_scan_candidates (
                id BIGSERIAL PRIMARY KEY,
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
                id BIGSERIAL PRIMARY KEY,
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
        sqlite_statements=(
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
    def __init__(self, db_path: Path | str, database_url: str | None = None):
        self.adapter: DatabaseAdapter = build_database_adapter(database_url, db_path)
        self.adapter.ensure_parent_dirs()

    def connection(self):
        return self.adapter.connection()

    def _ensure_migrations_table(self, conn) -> None:
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
            rows = conn.execute(self.adapter.prepare_sql("SELECT version FROM schema_migrations")).fetchall()
        return {int(row["version"]) for row in rows}

    def current_version(self) -> int:
        with self.connection() as conn:
            self._ensure_migrations_table(conn)
            row = conn.execute(
                self.adapter.prepare_sql("SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations")
            ).fetchone()
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
                for statement in migration.statements_for_backend(self.adapter.backend_name):
                    conn.execute(self.adapter.prepare_sql(statement))
                conn.execute(
                    self.adapter.prepare_sql(
                        """
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """
                    ),
                    (migration.version, migration.name),
                )
                applied_now.append(migration)
        return applied_now


def cli_main() -> None:
    settings = get_settings()
    runner = MigrationRunner(settings.db_path, settings.database_url)
    applied = runner.migrate()
    print(
        json.dumps(
            {
                "database_backend": runner.adapter.backend_name,
                "database_target": settings.database_url or str(settings.db_path),
                "current_version": runner.current_version(),
                "applied_migrations": [
                    {"version": migration.version, "name": migration.name}
                    for migration in applied
                ],
            }
        )
    )
