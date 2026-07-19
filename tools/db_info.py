#!/usr/bin/env python3

from database.connection import get_connection

with get_connection(read_only=True) as con:
    print("=== Tables ===")
    for row in con.execute("SHOW TABLES").fetchall():
        print(f" - {row[0]}")

    print("\n=== Row counts ===")
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")

    print("\n=== Database size ===")
    try:
        size = con.execute("PRAGMA database_size").fetchdf()
        print(size)
    except Exception as e:
        print(f"database_size unavailable: {e}")
