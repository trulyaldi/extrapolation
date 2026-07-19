import duckdb

from database.init_db import initialize_database


def test_initialize_fresh_database(tmp_path):
    db_path = tmp_path / "fresh.duckdb"

    initialize_database(db_path)
    initialize_database(db_path)

    assert db_path.exists()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert {"datasets", "dataset_columns", "dataset_rows", "dataset_cells"} <= tables


def test_initialize_migrates_legacy_result_database(tmp_path):
    db_path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("CREATE TABLE extrapolation_results (id BIGINT)")
        connection.execute("CREATE SEQUENCE extrapolation_result_id_seq")
        connection.execute(
            "CREATE TABLE schema_migrations (version VARCHAR PRIMARY KEY, applied_at TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES ('002_extrapolation_results.sql', CURRENT_TIMESTAMP)"
        )

    initialize_database(db_path)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert "extrapolation_results" not in tables
    assert "datasets" in tables
