#!/usr/bin/env python3
"""Command-line entry point for the DuckDB scientific source-data catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from database import DatasetDatabase, DatasetImportError, DatasetValidationError
from database.import_reference_values import import_reference_values


def _print_frame(frame) -> None:
    if frame.empty:
        print("(no rows)")
    else:
        print(frame.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="DuckDB file (defaults to EXTRAPOLATION_DB_PATH)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize the database and apply migrations")

    sync_parser = subparsers.add_parser("sync", help="import configured repository source files")
    sync_parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    sync_parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace changed imported data (normal sync never overwrites)",
    )
    sync_parser.add_argument(
        "--allow-schema-change",
        action="store_true",
        help="allow --replace to change a source dataset's column layout",
    )

    subparsers.add_parser("list", help="list datasets")

    metadata_parser = subparsers.add_parser("metadata", help="inspect dataset provenance and columns")
    metadata_parser.add_argument("dataset_name")

    export_parser = subparsers.add_parser("export", help="export one dataset as CSV")
    export_parser.add_argument("dataset_name")
    export_parser.add_argument("destination", type=Path)

    sql_parser = subparsers.add_parser("sql", help="run one read-only inspection query")
    sql_parser.add_argument("query")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db = DatasetDatabase(args.db)

    try:
        if args.command == "init":
            print(f"Database initialized: {db.initialize()}")
        elif args.command == "sync":
            db.initialize()
            report = db.sync_sources(
                args.source_root,
                replace=args.replace,
                allow_schema_change=args.allow_schema_change,
            )
            reference_files = [
                args.source_root / "large-dataset" / "reference_vals.txt",
                args.source_root / "src" / "new_reference_vals.txt",
            ]
            reference_rows = (
                import_reference_values(db.db_path, reference_files)
                if all(path.is_file() for path in reference_files)
                else 0
            )
            print(
                "sync: "
                f"imported={len(report.imported)}, replaced={len(report.replaced)}, "
                f"unchanged={len(report.unchanged)}, changed={len(report.changed)}, "
                f"schema_conflicts={len(report.schema_conflicts)}, "
                f"reference_rows_checked={reference_rows}"
            )
            for source_path in report.changed:
                print(f"changed (not overwritten): {source_path}")
            for source_path in report.schema_conflicts:
                print(f"schema changed (use --allow-schema-change): {source_path}")
        elif args.command == "list":
            _print_frame(db.list_datasets())
        elif args.command == "metadata":
            metadata = db.get_dataset_metadata(args.dataset_name)
            for key, value in metadata.items():
                print(f"{key}: {value}")
        elif args.command == "export":
            print(f"Exported: {db.export_dataset(args.dataset_name, args.destination)}")
        elif args.command == "sql":
            _print_frame(db.execute_readonly_sql(args.query))
    except (DatasetImportError, DatasetValidationError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
