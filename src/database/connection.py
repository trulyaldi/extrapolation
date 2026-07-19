"""DuckDB connection helpers for the scientific source-data catalog."""

import os
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(
    os.environ.get(
        "EXTRAPOLATION_DB_PATH",
        PROJECT_ROOT / "data" / "database" / "extrapolation.duckdb",
    )
)


def get_connection(
    read_only: bool = False, db_path: Path | str | None = None
) -> duckdb.DuckDBPyConnection:
    """Return a new connection; callers are responsible for closing it."""
    path = Path(db_path) if db_path is not None else DB_PATH
    return duckdb.connect(str(path), read_only=read_only)
