import sys
from pathlib import Path
from types import ModuleType

import pytest

from trading_agents.core.database import (
    DatabaseConfigurationError,
    PostgresAdapter,
    SqliteAdapter,
    build_database_adapter,
)
from trading_agents.core.migrations import MigrationRunner


def test_build_database_adapter_defaults_to_sqlite(tmp_path: Path):
    adapter = build_database_adapter(None, tmp_path / "trading.db")

    assert isinstance(adapter, SqliteAdapter)
    assert adapter.backend_name == "sqlite"


def test_build_database_adapter_uses_postgres_for_database_url(tmp_path: Path):
    adapter = build_database_adapter("postgresql://user:pass@localhost:5432/trading", tmp_path / "trading.db")

    assert isinstance(adapter, PostgresAdapter)
    assert adapter.backend_name == "postgresql"


def test_build_database_adapter_rejects_unsupported_database_url(tmp_path: Path):
    with pytest.raises(DatabaseConfigurationError, match="sqlite, postgres, or postgresql"):
        build_database_adapter("mysql://user:pass@localhost/db", tmp_path / "trading.db")


def test_postgres_adapter_rewrites_placeholders_and_uses_dict_rows(monkeypatch):
    connect_calls: list[dict] = []

    class FakeConnection:
        def __init__(self):
            self.closed = False
            self.committed = False
            self.executed: list[tuple[str, tuple[int, ...]]] = []

        def execute(self, sql: str, params=None):
            self.executed.append((sql, tuple(params or ())))
            return None

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    fake_connection = FakeConnection()
    fake_psycopg = ModuleType("psycopg")
    fake_rows = ModuleType("psycopg.rows")
    fake_rows.dict_row = object()

    def fake_connect(database_url: str, row_factory=None):
        connect_calls.append({"database_url": database_url, "row_factory": row_factory})
        return fake_connection

    fake_psycopg.connect = fake_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    adapter = PostgresAdapter("postgresql://user:pass@localhost:5432/trading")
    prepared_sql = adapter.prepare_sql("SELECT * FROM table WHERE id = ? AND status = ?")

    with adapter.connection() as conn:
        conn.execute(prepared_sql, (7, 1))

    assert prepared_sql == "SELECT * FROM table WHERE id = %s AND status = %s"
    assert connect_calls == [
        {
            "database_url": "postgresql://user:pass@localhost:5432/trading",
            "row_factory": fake_rows.dict_row,
        }
    ]
    assert fake_connection.executed == [(prepared_sql, (7, 1))]
    assert fake_connection.committed is True
    assert fake_connection.closed is True


def test_postgres_migration_runner_uses_postgres_specific_schema(monkeypatch, tmp_path: Path):
    executed_sql: list[str] = []

    class FakeCursor:
        def __init__(self, rows=None, row=None):
            self._rows = rows or []
            self._row = row

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._row

    class FakeConnection:
        def __init__(self):
            self.closed = False
            self.committed = False

        def execute(self, sql: str, params=None):
            executed_sql.append(sql.strip())
            normalized = " ".join(sql.split())
            if "SELECT version FROM schema_migrations" in normalized:
                return FakeCursor(rows=[])
            if "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations" in normalized:
                return FakeCursor(row={"version": 2})
            return FakeCursor()

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    fake_connection = FakeConnection()
    fake_psycopg = ModuleType("psycopg")
    fake_rows = ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    fake_psycopg.connect = lambda database_url, row_factory=None: fake_connection
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    runner = MigrationRunner(tmp_path / "trading.db", "postgresql://user:pass@localhost:5432/trading")
    applied = runner.migrate()

    assert [migration.version for migration in applied] == [1, 2]
    assert any("BIGSERIAL PRIMARY KEY" in sql for sql in executed_sql)
    assert not any("AUTOINCREMENT" in sql for sql in executed_sql)
