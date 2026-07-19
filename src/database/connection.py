from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "database" / "extrapolation.duckdb"

def get_connection(read_only=False):
    return duckdb.connect(str(DB_PATH), read_only=read_only)