"""Repository-root import bridge for the source-layout database package.

The implementation remains in ``src/database``.  Keeping this tiny package
makes ``from database import DatasetDatabase`` work when Python is launched
from the repository root without requiring a separately configured PYTHONPATH.
"""

from pathlib import Path


_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "database"
__path__.append(str(_SOURCE_PACKAGE))

from .datasets import (  # noqa: E402
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
