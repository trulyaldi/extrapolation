"""Package-qualified access to the DuckDB source-data API."""

from database import (
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

