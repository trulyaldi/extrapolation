import duckdb

from database.connection import DB_PATH, PROJECT_ROOT

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
migration_dir = PROJECT_ROOT / "sql" / "migrations"

with duckdb.connect(str(DB_PATH)) as con:
    for migration in sorted(migration_dir.glob("*.sql")):
        print(f"Applying {migration.name}...")
        con.execute(migration.read_text())

print(f"Database initialized: {DB_PATH}")
