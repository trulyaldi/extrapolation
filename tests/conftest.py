import os
from pathlib import Path

TEST_DB = Path("/tmp/extrapolation_test.duckdb")
os.environ["EXTRAPOLATION_DB_PATH"] = str(TEST_DB)

from database.init_db import initialize_database

if TEST_DB.exists():
    TEST_DB.unlink()

initialize_database(TEST_DB)
