from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "database" / "extrapolation.duckdb"
SOURCE_FILES = [
    ROOT / "large-dataset" / "reference_vals.txt",
    ROOT / "src" / "new_reference_vals.txt",
]


def read_rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    rows = []

    with path.open() as file:
        next(file)

        for line in file:
            if not line.strip():
                continue

            system, expectation_value, ref_value, uncertainty, source = line.split()
            rows.append(
                (system, expectation_value, ref_value, uncertainty, source)
            )

    return rows


def import_reference_values() -> None:
    with duckdb.connect(str(DB_PATH)) as connection:
        for source_file in SOURCE_FILES:
            connection.executemany(
                """
                INSERT OR IGNORE INTO reference_values
                VALUES (
                    ?,
                    ?,
                    CAST(? AS DECIMAL(38, 18)),
                    CAST(? AS DECIMAL(38, 18)),
                    ?
                )
                """,
                read_rows(source_file),
            )


if __name__ == "__main__":
    import_reference_values()
    print("Imported reference values.")
