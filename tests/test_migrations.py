import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from trading_agents.core.migrations import MIGRATIONS, MigrationRunner
from trading_agents.core.storage import Storage


def test_storage_runs_all_migrations_on_new_database(tmp_path: Path):
    db_path = tmp_path / "trading.db"

    storage = Storage(db_path)

    with storage.connection() as conn:
        versions = conn.execute("SELECT version, name FROM schema_migrations ORDER BY version ASC").fetchall()
        table_names = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        index_names = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        }

    assert storage.schema_version == len(MIGRATIONS)
    assert [(row["version"], row["name"]) for row in versions] == [
        (migration.version, migration.name) for migration in MIGRATIONS
    ]
    assert {
        "users",
        "signal_requests",
        "audit_log",
        "signal_events",
        "universe_scan_candidates",
        "opportunity_alpaca_orders",
        "schema_migrations",
    }.issubset(table_names)
    assert "idx_signal_requests_updated_at" in index_names
    assert "idx_signal_events_request_id_id" in index_names


def test_migrations_are_idempotent(tmp_path: Path):
    db_path = tmp_path / "trading.db"
    runner = MigrationRunner(db_path)

    first = runner.migrate()
    second = runner.migrate()

    assert [migration.version for migration in first] == [migration.version for migration in MIGRATIONS]
    assert second == []
    assert runner.current_version() == len(MIGRATIONS)

    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert row_count == len(MIGRATIONS)


def test_migration_cli_reports_current_version(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "trading.db"
    checkpoint_path = tmp_path / "langgraph-checkpoints.sqlite"
    chroma_dir = tmp_path / "chroma"

    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))

    result = subprocess.run(
        [sys.executable, "migrate_db.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["db_path"] == str(db_path)
    assert payload["current_version"] == len(MIGRATIONS)
    assert payload["applied_migrations"]
