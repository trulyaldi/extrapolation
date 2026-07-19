"""Command-line access to source data, fitting, and TOML studies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from database import DatasetDatabase, DatasetImportError, DatasetValidationError
from extrapolation.api import fit_all_observables, fit_dataset
from extrapolation.data import FitValidationError
from extrapolation.study import run_study


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_frame(frame: pd.DataFrame) -> None:
    print("(no rows)" if frame.empty else frame.to_string(index=False))


def _add_fit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default=None)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--n-fit", type=int)
    parser.add_argument("--basis-min", type=float)
    parser.add_argument("--basis-max", type=float)
    parser.add_argument(
        "--use-energy-b", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--no-uq", action="store_true")
    parser.add_argument("--missing", choices=("drop", "raise"), default="drop")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="extrapolate", description=__doc__)
    parser.add_argument("--db", type=Path, help="DuckDB file")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="manage the DuckDB source-data catalog")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("init")
    sync = data_commands.add_parser("sync")
    sync.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    sync.add_argument("--replace", action="store_true")
    sync.add_argument("--allow-schema-change", action="store_true")
    data_commands.add_parser("list")
    show = data_commands.add_parser("show")
    show.add_argument("dataset")
    export = data_commands.add_parser("export")
    export.add_argument("dataset")
    export.add_argument("destination", type=Path)

    fit = commands.add_parser("fit", help="fit one dataset")
    fit.add_argument("dataset")
    target = fit.add_mutually_exclusive_group(required=True)
    target.add_argument("--observable")
    target.add_argument("--all", action="store_true", dest="all_observables")
    _add_fit_options(fit)
    fit.add_argument("--json", type=Path)
    fit.add_argument("--csv", type=Path)
    fit.add_argument("--plot-dir", type=Path)

    run = commands.add_parser("run", help="run a TOML study")
    run.add_argument("study", type=Path)
    return parser


def _fit_options(args: argparse.Namespace) -> dict:
    return {
        "method": args.method,
        "model": args.model,
        "n_fit": args.n_fit,
        "basis_min": args.basis_min,
        "basis_max": args.basis_max,
        "use_energy_b": args.use_energy_b,
        "compute_uq": not args.no_uq,
        "missing": args.missing,
        "verbose": args.verbose,
    }


def _export_cli_results(args: argparse.Namespace, results: dict) -> None:
    output_files: list[str] = []
    manifest_path: Path | None = None
    if args.json:
        if len(results) == 1:
            next(iter(results.values())).to_json(args.json)
        else:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    {name: result.to_dict() for name, result in results.items()},
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        output_files.append(str(args.json))
        manifest_path = args.json.with_name(f"{args.json.stem}.manifest.json")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "dataset": result.dataset_name,
                    "observable": result.observable,
                    "model": result.model,
                    "baseline": result.baseline,
                    "baseline_uncertainty": result.baseline_uncertainty,
                    "r_squared": result.r_squared,
                }
                for result in results.values()
            ]
        ).to_csv(args.csv, index=False)
        output_files.append(str(args.csv))
        if manifest_path is None:
            manifest_path = args.csv.with_name(f"{args.csv.stem}.manifest.json")
    if args.plot_dir:
        import matplotlib.pyplot as plt

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            stem = name.replace("/", "_").replace(":", "_").replace(" ", "_")
            for suffix, plotter in (
                ("fit", result.plot),
                ("log", result.plot_log),
                ("profile", result.plot_profile),
            ):
                plot_path = args.plot_dir / f"{stem}_{suffix}.png"
                figure, _ = plotter(output_path=plot_path)
                plt.close(figure)
                output_files.append(str(plot_path))
        if manifest_path is None:
            manifest_path = args.plot_dir / "run.manifest.json"
    if manifest_path is not None:
        output_files.append(str(manifest_path))
        if len(results) == 1:
            manifest = next(iter(results.values())).manifest(output_files)
        else:
            from extrapolation import __version__

            manifest = {
                "manifest_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "package_version": __version__,
                "runs": [result.manifest() for result in results.values()],
                "output_files": output_files,
            }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db = DatasetDatabase(args.db)
    try:
        if args.command == "data":
            if args.data_command == "init":
                print(f"Database initialized: {db.initialize()}")
            elif args.data_command == "sync":
                db.initialize()
                report = db.sync_sources(
                    args.source_root,
                    replace=args.replace,
                    allow_schema_change=args.allow_schema_change,
                )
                print(
                    f"sync: imported={len(report.imported)}, "
                    f"replaced={len(report.replaced)}, unchanged={len(report.unchanged)}, "
                    f"changed={len(report.changed)}, "
                    f"schema_conflicts={len(report.schema_conflicts)}"
                )
            elif args.data_command == "list":
                _print_frame(db.list_datasets())
            elif args.data_command == "show":
                metadata = db.get_dataset_metadata(args.dataset)
                for key, value in metadata.items():
                    print(f"{key}: {value}")
            elif args.data_command == "export":
                print(f"Exported: {db.export_dataset(args.dataset, args.destination)}")
        elif args.command == "fit":
            options = _fit_options(args)
            if args.all_observables:
                results = fit_all_observables(
                    args.dataset, db=db, on_error="skip", **options
                )
                if not results:
                    raise FitValidationError("No observables could be fitted.")
            else:
                result = fit_dataset(
                    args.dataset, args.observable, db=db, **options
                )
                results = {args.observable: result}
            for result in results.values():
                print(result.summary())
            failures = next(iter(results.values())).metadata.get("batch_failures", {})
            for observable, message in failures.items():
                print(f"skipped {observable}: {message}", file=sys.stderr)
            _export_cli_results(args, results)
        elif args.command == "run":
            study = run_study(args.study, db=db)
            print(
                f"study={study.name}, fits={len(study.results)}, "
                f"manifest={study.manifest_path}"
            )
    except (
        DatasetImportError,
        DatasetValidationError,
        FitValidationError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
