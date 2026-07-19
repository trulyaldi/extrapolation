"""DuckDB-backed source scientific-data API."""

from database.datasets import (
    DatasetDatabase,
    DatasetImportError,
    DatasetValidationError,
    SyncReport,
    dataset_name_for_source,
    discover_source_files,
)

__all__ = [
    "DatasetDatabase",
    "DatasetImportError",
    "DatasetValidationError",
    "SyncReport",
    "dataset_name_for_source",
    "discover_source_files",
]
