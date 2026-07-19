import duckdb

from database.connection import DB_PATH, PROJECT_ROOT

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
migration_dir = PROJECT_ROOT / "sql" / "migrations"

with duckdb.connect(str(DB_PATH)) as con:
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

        print(f"Applying {migration.name}...")
        con.execute(migration.read_text())
        con.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            [migration.name],
        )

print(f"Database initialized: {DB_PATH}")
