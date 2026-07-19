from pathlib import Path
from typing import Iterable

import duckdb

from database.connection import DB_PATH, PROJECT_ROOT


SOURCE_FILES = (
    PROJECT_ROOT / "large-dataset" / "reference_vals.txt",
    PROJECT_ROOT / "src" / "new_reference_vals.txt",
)


def read_rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    rows = []

    with path.open() as file:
        next(file)

        for line in file:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{path}: expected five whitespace-delimited values on line {len(rows) + 2}"
                )
            system, expectation_value, ref_value, uncertainty, source = parts
            rows.append(
                (system, expectation_value, ref_value, uncertainty, source)
            )

    return rows


def import_reference_values(
    db_path: Path | str = DB_PATH,
    source_files: Iterable[Path] = SOURCE_FILES,
) -> int:
    """Import bundled reference values without overwriting catalog entries.

    Reference data are input values, not extrapolation outputs.  ``INSERT OR
    IGNORE`` preserves an existing catalog value if this command is repeated.
    The destination database must already be initialized.
    """
    rows = [row for source_file in source_files for row in read_rows(source_file)]
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.executemany(
                """
                INSERT OR IGNORE INTO reference_values
                VALUES (
                    ?,
                    ?,
                    CAST(? AS DECIMAL(38, 30)),
                    CAST(? AS DECIMAL(38, 30)),
                    ?
                )
                """,
                rows,
            )
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
    return len(rows)


if __name__ == "__main__":
    import_reference_values()
    print("Imported reference values.")
