from pathlib import Path
import duckdb

db_path = Path("data/database/extrapolation.duckdb")
db_path.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))

migration_dir = Path("sql/migrations")

for migration in sorted(migration_dir.glob("*.sql")):
    print(f"Applying {migration.name}...")
    con.execute(migration.read_text())

con.close()

print(f"Database initialized: {db_path}")
