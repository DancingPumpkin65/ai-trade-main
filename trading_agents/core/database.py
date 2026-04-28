from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


class DatabaseConfigurationError(RuntimeError):
    pass


class DatabaseDependencyError(RuntimeError):
    pass


class DatabaseAdapter:
    backend_name = "unknown"

    def __init__(self, target: str | Path):
        self.target = target

    @contextmanager
    def connection(self) -> Iterator[Any]:
        raise NotImplementedError

    def prepare_sql(self, sql: str) -> str:
        return sql

    def ensure_parent_dirs(self) -> None:
        return None


class SqliteAdapter(DatabaseAdapter):
    backend_name = "sqlite"

    def __init__(self, target: str | Path):
        db_path = Path(target)
        super().__init__(db_path)
        self.db_path = db_path

    def ensure_parent_dirs(self) -> None:
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


class PostgresAdapter(DatabaseAdapter):
    backend_name = "postgresql"

    def __init__(self, target: str):
        super().__init__(target)
        self.database_url = target

    def _load_psycopg(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on optional runtime install
            raise DatabaseDependencyError(
                "PostgreSQL support requires the 'psycopg' package to be installed."
            ) from exc
        return psycopg, dict_row

    def prepare_sql(self, sql: str) -> str:
        return sql.replace("?", "%s")

    @contextmanager
    def connection(self) -> Iterator[Any]:
        psycopg, dict_row = self._load_psycopg()
        conn = psycopg.connect(self.database_url, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def is_postgres_url(value: str | None) -> bool:
    if not value:
        return False
    scheme = urlparse(value).scheme.lower()
    return scheme in {"postgres", "postgresql"}


def build_database_adapter(database_url: str | None, db_path: str | Path) -> DatabaseAdapter:
    if is_postgres_url(database_url):
        return PostgresAdapter(database_url)
    if database_url and urlparse(database_url).scheme and not is_postgres_url(database_url):
        raise DatabaseConfigurationError(
            "DATABASE_URL must use a sqlite, postgres, or postgresql scheme."
        )
    return SqliteAdapter(Path(db_path))
