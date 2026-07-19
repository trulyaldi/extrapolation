"""Scientific source-data storage, import, and editing APIs.

The catalog keeps CSV layouts in a normalized set of metadata, row, and cell
tables.  That avoids an unbounded set of per-file SQL tables while allowing
``load_dataset`` to return the original wide, ordered pandas DataFrame.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pandas as pd

from database.connection import DB_PATH, PROJECT_ROOT, get_connection
from database.init_db import initialize_database


DATA_DIRECTORIES = (
    "large-dataset",
    "large-dataset-init",
    "large-dataset-err",
    "new-dataset",
    "small-dataset",
    "muon",
)
TEXT_DATA_SUFFIXES = {".csv", ".xls"}
DATA_ROLES = {"observation", "uncertainty", "auxiliary"}
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]*$")


class DatasetValidationError(ValueError):
    """Raised when a dataset name, shape, or numeric value is invalid."""


class DatasetImportError(ValueError):
    """Raised when a configured source file cannot be imported safely."""


@dataclass(frozen=True)
class SyncReport:
    """Summary of a source synchronization attempt."""

    imported: tuple[str, ...] = ()
    replaced: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    schema_conflicts: tuple[str, ...] = ()


def _validate_dataset_name(dataset_name: str) -> str:
    if not isinstance(dataset_name, str) or not _DATASET_NAME_RE.fullmatch(dataset_name):
        raise DatasetValidationError(
            "dataset_name must contain only letters, numbers, '/', '+', '-', '_', '.', "
            "or ':' and cannot start with punctuation"
        )
    if len(dataset_name) > 240:
        raise DatasetValidationError("dataset_name is too long")
    return dataset_name


def _validate_column_name(column_name: str) -> str:
    if not isinstance(column_name, str) or not column_name or not column_name.strip():
        raise DatasetValidationError("column names must be non-empty strings")
    if "\x00" in column_name:
        raise DatasetValidationError("column names cannot contain NUL characters")
    return column_name


def _validate_role(data_role: str) -> str:
    if data_role not in DATA_ROLES:
        allowed = ", ".join(sorted(DATA_ROLES))
        raise DatasetValidationError(f"data_role must be one of: {allowed}")
    return data_role


def _validate_row_index(row_index: int) -> int:
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise DatasetValidationError("row_index must be a non-negative integer")
    return row_index


def _coerce_numeric(value: Any, context: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        raise DatasetValidationError(f"{context} must be numeric, not boolean")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"{context} must be numeric or missing") from error
    if not math.isfinite(numeric_value):
        raise DatasetValidationError(f"{context} must be finite when present")
    return numeric_value


def _dataframe_rows(data: pd.DataFrame) -> tuple[list[str], list[list[float | None]]]:
    if not isinstance(data, pd.DataFrame):
        raise DatasetValidationError("data must be a pandas DataFrame")
    columns = [_validate_column_name(str(column)) for column in data.columns]
    if not columns:
        raise DatasetValidationError("a dataset needs at least one column")
    if len(set(columns)) != len(columns):
        raise DatasetValidationError("dataset column names must be unique")

    rows: list[list[float | None]] = []
    for row_index, row in enumerate(data.itertuples(index=False, name=None)):
        rows.append(
            [
                _coerce_numeric(value, f"row {row_index}, column {columns[column_index]!r}")
                for column_index, value in enumerate(row)
            ]
        )
    return columns, rows


def _schema_signature(columns: Sequence[str]) -> str:
    return json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_file(path: Path) -> pd.DataFrame:
    """Read one of the repository's comma-delimited source files.

    Some historical files use an ``.xls`` suffix but contain CSV text, so the
    suffix is deliberately not used to choose a spreadsheet reader.
    """
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                columns = next(reader)
            except StopIteration as error:
                raise DatasetImportError(f"{path}: source file is empty") from error

            columns = [_validate_column_name(column) for column in columns]
            if len(set(columns)) != len(columns):
                raise DatasetImportError(f"{path}: duplicate column names are not supported")
            if not any(column.strip().casefold() == "basis size" for column in columns):
                raise DatasetImportError(
                    f"{path}: expected a 'Basis Size' independent-variable column"
                )

            rows: list[list[float | None]] = []
            for source_row_number, raw_row in enumerate(reader, start=2):
                if not raw_row or all(not cell.strip() for cell in raw_row):
                    continue
                if len(raw_row) != len(columns):
                    raise DatasetImportError(
                        f"{path}: row {source_row_number} has {len(raw_row)} cells; "
                        f"expected {len(columns)}"
                    )
                values: list[float | None] = []
                for column_name, raw_value in zip(columns, raw_row):
                    if not raw_value.strip():
                        values.append(None)
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError as error:
                        raise DatasetImportError(
                            f"{path}: row {source_row_number}, column {column_name!r} "
                            f"contains non-numeric value {raw_value!r}"
                        ) from error
                    if not math.isfinite(value):
                        raise DatasetImportError(
                            f"{path}: row {source_row_number}, column {column_name!r} "
                            "must be finite when present"
                        )
                    values.append(value)
                rows.append(values)
    except OSError as error:
        raise DatasetImportError(f"Could not read {path}: {error}") from error

    return pd.DataFrame(rows, columns=columns)


def discover_source_files(source_root: Path | str = PROJECT_ROOT) -> list[Path]:
    """Return configured scientific CSV sources in deterministic path order."""
    root = Path(source_root)
    files: list[Path] = []
    for directory_name in DATA_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in TEXT_DATA_SUFFIXES
        )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def dataset_name_for_source(path: Path | str, source_root: Path | str = PROJECT_ROOT) -> str:
    """Derive a stable logical dataset name from a repository-relative path."""
    source_path = Path(path)
    root = Path(source_root).resolve()
    try:
        relative = source_path.resolve().relative_to(root)
    except ValueError as error:
        raise DatasetValidationError(
            f"source file {source_path} is outside configured root {root}"
        ) from error
    return _validate_dataset_name(relative.with_suffix("").as_posix().casefold())


def _role_for_source(path: Path, source_root: Path) -> str:
    relative_parts = path.resolve().relative_to(source_root.resolve()).parts
    return "uncertainty" if relative_parts[0].casefold() == "large-dataset-err" else "observation"


class DatasetDatabase:
    """Small, transaction-safe API for source scientific data.

    Call :meth:`initialize` once for a new database.  The class has no global
    connection and does not initialize or mutate a database at import time.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH

    def initialize(self) -> Path:
        """Create the database if needed and apply pending migrations."""
        return initialize_database(self.db_path)

    @contextmanager
    def _read_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self.db_path.exists():
            raise RuntimeError(
                f"Database does not exist at {self.db_path}. Call db.initialize() first."
            )
        connection = get_connection(read_only=True, db_path=self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self.db_path.exists():
            raise RuntimeError(
                f"Database does not exist at {self.db_path}. Call db.initialize() first."
            )
        connection = get_connection(db_path=self.db_path)
        connection.execute("BEGIN TRANSACTION")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
        finally:
            connection.close()

    @staticmethod
    def _dataset_exists(connection: duckdb.DuckDBPyConnection, dataset_name: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM datasets WHERE dataset_name = ?", [dataset_name]
        ).fetchone() is not None

    @staticmethod
    def _columns_for(
        connection: duckdb.DuckDBPyConnection, dataset_name: str
    ) -> list[str]:
        return [
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM dataset_columns
                WHERE dataset_name = ?
                ORDER BY column_index
                """,
                [dataset_name],
            ).fetchall()
        ]

    @staticmethod
    def _replace_dataset(
        connection: duckdb.DuckDBPyConnection,
        dataset_name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[float | None]],
        *,
        data_role: str,
        independent_column: str,
        source_path: str | None,
        source_hash: str | None,
        manual_modified: bool,
    ) -> None:
        if source_path is not None:
            conflicting = connection.execute(
                """
                SELECT dataset_name FROM datasets
                WHERE source_path = ? AND dataset_name != ?
                """,
                [source_path, dataset_name],
            ).fetchone()
            if conflicting is not None:
                raise DatasetValidationError(
                    f"source path {source_path!r} already belongs to {conflicting[0]!r}"
                )

        for table_name in ("dataset_cells", "dataset_rows", "dataset_columns", "datasets"):
            connection.execute(f"DELETE FROM {table_name} WHERE dataset_name = ?", [dataset_name])

        connection.execute(
            """
            INSERT INTO datasets (
                dataset_name, data_role, source_path, source_hash,
                independent_column, schema_signature, row_count,
                manual_modified, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                      CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END)
            """,
            [
                dataset_name,
                data_role,
                source_path,
                source_hash,
                independent_column,
                _schema_signature(columns),
                len(rows),
                manual_modified,
                source_path,
            ],
        )
        connection.executemany(
            """
            INSERT INTO dataset_columns (dataset_name, column_index, column_name)
            VALUES (?, ?, ?)
            """,
            [(dataset_name, index, column) for index, column in enumerate(columns)],
        )
        if rows:
            connection.executemany(
                "INSERT INTO dataset_rows (dataset_name, row_index) VALUES (?, ?)",
                [(dataset_name, row_index) for row_index in range(len(rows))],
            )
            cells = [
                (dataset_name, row_index, column, value)
                for row_index, row in enumerate(rows)
                for column, value in zip(columns, row)
            ]
            connection.executemany(
                """
                INSERT INTO dataset_cells (dataset_name, row_index, column_name, numeric_value)
                VALUES (?, ?, ?, ?)
                """,
                cells,
            )

    def list_datasets(self) -> pd.DataFrame:
        """List available datasets and concise provenance/editing metadata."""
        with self._read_connection() as connection:
            return connection.execute(
                """
                SELECT
                    dataset_name,
                    data_role,
                    source_path,
                    row_count,
                    independent_column,
                    manual_modified,
                    imported_at,
                    updated_at
                FROM datasets
                ORDER BY dataset_name
                """
            ).fetchdf()

    def get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]:
        """Return provenance, shape, role, and ordered original column names."""
        dataset_name = _validate_dataset_name(dataset_name)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    dataset_name, data_role, source_path, source_hash,
                    independent_column, schema_signature, row_count,
                    manual_modified, created_at, updated_at, imported_at
                FROM datasets
                WHERE dataset_name = ?
                """,
                [dataset_name],
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown dataset {dataset_name!r}")
            columns = self._columns_for(connection, dataset_name)

        return {
            "dataset_name": row[0],
            "data_role": row[1],
            "source_path": row[2],
            "source_hash": row[3],
            "independent_column": row[4],
            "schema_signature": row[5],
            "row_count": row[6],
            "manual_modified": row[7],
            "created_at": row[8],
            "updated_at": row[9],
            "imported_at": row[10],
            "columns": columns,
        }

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load one dataset as a fitting-ready wide DataFrame in row order."""
        dataset_name = _validate_dataset_name(dataset_name)
        with self._read_connection() as connection:
            if not self._dataset_exists(connection, dataset_name):
                raise KeyError(f"Unknown dataset {dataset_name!r}")
            columns = self._columns_for(connection, dataset_name)
            row_indices = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT row_index FROM dataset_rows
                    WHERE dataset_name = ?
                    ORDER BY row_index
                    """,
                    [dataset_name],
                ).fetchall()
            ]
            values_by_row = {row_index: {} for row_index in row_indices}
            for row_index, column_name, numeric_value in connection.execute(
                """
                SELECT row_index, column_name, numeric_value
                FROM dataset_cells
                WHERE dataset_name = ?
                ORDER BY row_index, column_name
                """,
                [dataset_name],
            ).fetchall():
                values_by_row[row_index][column_name] = numeric_value

        rows = [
            [values_by_row[row_index].get(column_name) for column_name in columns]
            for row_index in row_indices
        ]
        return pd.DataFrame(rows, columns=columns)

    def load_observations(self, dataset_name: str) -> pd.DataFrame:
        """Load an observation dataset, rejecting uncertainty-only datasets."""
        metadata = self.get_dataset_metadata(dataset_name)
        if metadata["data_role"] != "observation":
            raise DatasetValidationError(
                f"{dataset_name!r} is {metadata['data_role']!r}, not an observation dataset"
            )
        return self.load_dataset(dataset_name)

    def add_or_replace_dataset(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        *,
        data_role: str = "observation",
        independent_column: str | None = None,
        replace: bool = False,
    ) -> None:
        """Add a manual dataset, or explicitly replace one with a DataFrame."""
        dataset_name = _validate_dataset_name(dataset_name)
        data_role = _validate_role(data_role)
        columns, rows = _dataframe_rows(data)
        independent_column = _validate_column_name(independent_column or columns[0])
        if independent_column not in columns:
            raise DatasetValidationError("independent_column must be one of the DataFrame columns")

        with self._write_transaction() as connection:
            if self._dataset_exists(connection, dataset_name) and not replace:
                raise DatasetValidationError(
                    f"Dataset {dataset_name!r} already exists; pass replace=True to replace it"
                )
            self._replace_dataset(
                connection,
                dataset_name,
                columns,
                rows,
                data_role=data_role,
                independent_column=independent_column,
                source_path=None,
                source_hash=None,
                manual_modified=True,
            )

    def update_value(
        self, dataset_name: str, row_index: int, column_name: str, value: float | None
    ) -> None:
        """Update one numeric or missing value and mark the dataset as manually edited."""
        dataset_name = _validate_dataset_name(dataset_name)
        row_index = _validate_row_index(row_index)
        column_name = _validate_column_name(column_name)
        numeric_value = _coerce_numeric(value, "value")

        with self._write_transaction() as connection:
            if not self._dataset_exists(connection, dataset_name):
                raise KeyError(f"Unknown dataset {dataset_name!r}")
            if column_name not in self._columns_for(connection, dataset_name):
                raise KeyError(f"Unknown column {column_name!r} for {dataset_name!r}")
            row_exists = connection.execute(
                "SELECT 1 FROM dataset_rows WHERE dataset_name = ? AND row_index = ?",
                [dataset_name, row_index],
            ).fetchone()
            if row_exists is None:
                raise KeyError(f"Unknown row {row_index} for {dataset_name!r}")
            connection.execute(
                """
                UPDATE dataset_cells
                SET numeric_value = ?
                WHERE dataset_name = ? AND row_index = ? AND column_name = ?
                """,
                [numeric_value, dataset_name, row_index, column_name],
            )
            connection.execute(
                """
                UPDATE datasets
                SET manual_modified = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE dataset_name = ?
                """,
                [dataset_name],
            )

    def insert_row(
        self,
        dataset_name: str,
        values: Mapping[str, float | None],
        *,
        row_index: int | None = None,
    ) -> int:
        """Append a row (or add it at an unused explicit row index) transactionally."""
        dataset_name = _validate_dataset_name(dataset_name)
        if not isinstance(values, Mapping):
            raise DatasetValidationError("values must map column names to numeric values")

        with self._write_transaction() as connection:
            if not self._dataset_exists(connection, dataset_name):
                raise KeyError(f"Unknown dataset {dataset_name!r}")
            columns = self._columns_for(connection, dataset_name)
            unknown_columns = set(values) - set(columns)
            if unknown_columns:
                raise KeyError(f"Unknown columns for {dataset_name!r}: {sorted(unknown_columns)!r}")
            if row_index is None:
                row_index = connection.execute(
                    """
                    SELECT COALESCE(MAX(row_index) + 1, 0)
                    FROM dataset_rows WHERE dataset_name = ?
                    """,
                    [dataset_name],
                ).fetchone()[0]
            row_index = _validate_row_index(row_index)
            if connection.execute(
                "SELECT 1 FROM dataset_rows WHERE dataset_name = ? AND row_index = ?",
                [dataset_name, row_index],
            ).fetchone() is not None:
                raise DatasetValidationError(
                    f"row_index {row_index} already exists for {dataset_name!r}"
                )

            cell_values = [
                _coerce_numeric(values.get(column_name), f"column {column_name!r}")
                for column_name in columns
            ]
            connection.execute(
                "INSERT INTO dataset_rows (dataset_name, row_index) VALUES (?, ?)",
                [dataset_name, row_index],
            )
            connection.executemany(
                """
                INSERT INTO dataset_cells (dataset_name, row_index, column_name, numeric_value)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (dataset_name, row_index, column_name, numeric_value)
                    for column_name, numeric_value in zip(columns, cell_values)
                ],
            )
            connection.execute(
                """
                UPDATE datasets
                SET row_count = row_count + 1,
                    manual_modified = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE dataset_name = ?
                """,
                [dataset_name],
            )
        return row_index

    def delete_row(self, dataset_name: str, row_index: int) -> None:
        """Delete a row without renumbering the remaining ordered row indexes."""
        dataset_name = _validate_dataset_name(dataset_name)
        row_index = _validate_row_index(row_index)
        with self._write_transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM dataset_rows WHERE dataset_name = ? AND row_index = ?",
                [dataset_name, row_index],
            ).fetchone() is None:
                raise KeyError(f"Unknown row {row_index} for {dataset_name!r}")
            connection.execute(
                "DELETE FROM dataset_cells WHERE dataset_name = ? AND row_index = ?",
                [dataset_name, row_index],
            )
            connection.execute(
                "DELETE FROM dataset_rows WHERE dataset_name = ? AND row_index = ?",
                [dataset_name, row_index],
            )
            connection.execute(
                """
                UPDATE datasets
                SET row_count = row_count - 1,
                    manual_modified = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE dataset_name = ?
                """,
                [dataset_name],
            )

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete one explicitly named dataset and all of its catalog cells."""
        dataset_name = _validate_dataset_name(dataset_name)
        with self._write_transaction() as connection:
            if not self._dataset_exists(connection, dataset_name):
                raise KeyError(f"Unknown dataset {dataset_name!r}")
            for table_name in ("dataset_cells", "dataset_rows", "dataset_columns", "datasets"):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_name = ?", [dataset_name])

    def export_dataset(self, dataset_name: str, destination: Path | str) -> Path:
        """Export a dataset as a wide CSV file with original columns and order."""
        dataset_name = _validate_dataset_name(dataset_name)
        destination = Path(destination)
        data = self.load_dataset(dataset_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(data.columns)
            for row in data.itertuples(index=False, name=None):
                writer.writerow("" if pd.isna(value) else value for value in row)
        return destination

    def execute_readonly_sql(
        self, sql: str, parameters: Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Run one read-only SQL query for advanced inspection.

        DuckDB's read-only connection enforces the policy.  The small lexical
        check makes accidental write statements and multi-statement input clear
        to users before they reach the database.
        """
        if not isinstance(sql, str):
            raise DatasetValidationError("sql must be a string")
        statement = sql.strip()
        if not statement or ";" in statement:
            raise DatasetValidationError("execute_readonly_sql accepts one query without ';'")
        if not re.match(r"^(SELECT|WITH|EXPLAIN)\b", statement, flags=re.IGNORECASE):
            raise DatasetValidationError("only SELECT, WITH, and EXPLAIN queries are allowed")
        with self._read_connection() as connection:
            return connection.execute(statement, parameters or []).fetchdf()

    def sync_sources(
        self,
        source_root: Path | str = PROJECT_ROOT,
        *,
        replace: bool = False,
        allow_schema_change: bool = False,
    ) -> SyncReport:
        """Import configured source files in a single transaction.

        Normal sync is database-authoritative: same-hash sources are skipped
        and changed sources are reported without replacing database data.
        ``replace=True`` explicitly replaces changed source values; a changed
        layout additionally requires ``allow_schema_change=True``.
        """
        source_root = Path(source_root).resolve()
        source_files = discover_source_files(source_root)
        imported: list[str] = []
        replaced: list[str] = []
        unchanged: list[str] = []
        changed: list[str] = []
        schema_conflicts: list[str] = []

        with self._write_transaction() as connection:
            for source_file in source_files:
                source_path = source_file.relative_to(source_root).as_posix()
                dataset_name = dataset_name_for_source(source_file, source_root)
                source_hash = _hash_file(source_file)
                existing = connection.execute(
                    """
                    SELECT dataset_name, source_hash
                    FROM datasets WHERE source_path = ?
                    """,
                    [source_path],
                ).fetchone()

                if existing is not None and existing[1] == source_hash:
                    unchanged.append(source_path)
                    continue
                if existing is not None and not replace:
                    changed.append(source_path)
                    continue

                data = _read_source_file(source_file)
                columns, rows = _dataframe_rows(data)
                independent_column = next(
                    column for column in columns if column.strip().casefold() == "basis size"
                )
                if existing is not None:
                    current_columns = self._columns_for(connection, existing[0])
                    if current_columns != columns and not allow_schema_change:
                        schema_conflicts.append(source_path)
                        continue
                    if existing[0] != dataset_name:
                        raise DatasetImportError(
                            f"{source_file}: source path already belongs to "
                            f"dataset {existing[0]!r}"
                        )

                self._replace_dataset(
                    connection,
                    dataset_name,
                    columns,
                    rows,
                    data_role=_role_for_source(source_file, source_root),
                    independent_column=independent_column,
                    source_path=source_path,
                    source_hash=source_hash,
                    manual_modified=False,
                )
                (replaced if existing is not None else imported).append(source_path)

        return SyncReport(
            imported=tuple(imported),
            replaced=tuple(replaced),
            unchanged=tuple(unchanged),
            changed=tuple(changed),
            schema_conflicts=tuple(schema_conflicts),
        )
