from database.init_db import initialize_database


def test_initialize_fresh_database(tmp_path):
    db_path = tmp_path / "fresh.duckdb"

    initialize_database(db_path)
    initialize_database(db_path)

    assert db_path.exists()
