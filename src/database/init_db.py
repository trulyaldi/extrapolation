from pathlib import Path

import duckdb

from database.connection import DB_PATH, PROJECT_ROOT


def initialize_database(db_path: Path | str = DB_PATH) -> Path:
    """Apply all pending schema migrations to *db_path*.

    Initialization is explicit so importing the database package never mutates
    a local database as a side effect.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migration_dir = PROJECT_ROOT / "sql" / "migrations"

    with duckdb.connect(str(db_path)) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            applied = {
                row[0]
                for row in con.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }

            for migration in sorted(migration_dir.glob("*.sql")):
                if migration.name in applied:
                    continue
                try:
                    con.execute(migration.read_text())
                except duckdb.Error as error:
                    raise RuntimeError(
                        f"Could not apply migration {migration.name}: {error}"
                    ) from error
                con.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    [migration.name],
                )
        except BaseException:
            con.execute("ROLLBACK")
            raise
        else:
            con.execute("COMMIT")

    return db_path


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DB_PATH}")
