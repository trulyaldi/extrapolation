import duckdb

from database.connection import DB_PATH, PROJECT_ROOT


def initialize_database(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migration_dir = PROJECT_ROOT / "sql" / "migrations"

    with duckdb.connect(str(db_path)) as con:
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

            con.execute(migration.read_text())
            con.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)",
                [migration.name],
            )


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DB_PATH}")
