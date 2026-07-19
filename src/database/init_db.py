from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "database" / "extrapolation.duckdb"
MIGRATION_PATH = ROOT / "sql" / "migrations" / "001_initial_schema.sql"


def initialize_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    migration_sql = MIGRATION_PATH.read_text()

    with duckdb.connect(str(DB_PATH)) as connection:
        connection.execute(migration_sql)


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized database at {DB_PATH}")
