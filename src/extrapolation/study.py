"""Minimal TOML study runner built on the public fitting API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any

import pandas as pd

from database import DatasetDatabase
from extrapolation.api import fit_dataset
from extrapolation.result import FitResult


_TOP_KEYS = {"study", "datasets", "observables", "fit", "outputs"}
_FIT_KEYS = {
    "method",
    "model",
    "n_fit",
    "basis_min",
    "basis_max",
    "use_energy_b",
    "compute_uq",
    "missing",
    "verbose",
}
_OUTPUT_KEYS = {"directory", "summary_csv", "result_json", "plots"}


@dataclass(frozen=True)
class StudyResult:
    name: str
    results: dict[str, FitResult]
    output_directory: Path
    summary_path: Path | None
    manifest_path: Path


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{key} must be a nonempty array of strings")
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() != ".toml":
        raise ValueError("Study files must use the .toml format")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    _reject_unknown(config, _TOP_KEYS, "top-level")
    name = config.get("study")
    if not isinstance(name, str) or not name:
        raise ValueError("study must be a nonempty string")
    _string_list(config.get("datasets"), "datasets")
    _string_list(config.get("observables"), "observables")
    fit = config.get("fit", {})
    outputs = config.get("outputs", {})
    if not isinstance(fit, dict) or not isinstance(outputs, dict):
        raise ValueError("fit and outputs must be TOML tables")
    _reject_unknown(fit, _FIT_KEYS, "fit")
    _reject_unknown(outputs, _OUTPUT_KEYS, "outputs")
    if "n_fit" in fit and (not isinstance(fit["n_fit"], int) or fit["n_fit"] < 3):
        raise ValueError("fit.n_fit must be an integer of at least 3")
    for key in ("method", "model", "missing"):
        if key in fit and not isinstance(fit[key], str):
            raise ValueError(f"fit.{key} must be a string")
    for key in ("basis_min", "basis_max"):
        if key in fit and (
            isinstance(fit[key], bool)
            or not isinstance(fit[key], (int, float))
            or not math.isfinite(fit[key])
        ):
            raise ValueError(f"fit.{key} must be a finite number")
    for key in ("use_energy_b", "compute_uq", "verbose"):
        if key in fit and not isinstance(fit[key], bool):
            raise ValueError(f"fit.{key} must be true or false")
    for key in ("directory", "summary_csv"):
        if key in outputs and not isinstance(outputs[key], str):
            raise ValueError(f"outputs.{key} must be a string")
    for key in ("result_json", "plots"):
        if key in outputs and not isinstance(outputs[key], bool):
            raise ValueError(f"outputs.{key} must be true or false")
    return config


def run_study(path: str | Path, *, db: DatasetDatabase | None = None) -> StudyResult:
    """Execute a validated TOML study using :func:`fit_dataset` exclusively."""
    config_path = Path(path).resolve()
    config = _load_config(config_path)
    db = db or DatasetDatabase()
    fit_options = dict(config.get("fit", {}))
    outputs = dict(config.get("outputs", {}))
    output_directory = Path(outputs.get("directory", f"outputs/{config['study']}"))
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, FitResult] = {}
    output_files: list[str] = []
    for dataset in config["datasets"]:
        for observable in config["observables"]:
            result = fit_dataset(dataset, observable, db=db, **fit_options)
            key = f"{dataset}:{observable}"
            results[key] = result
            if outputs.get("result_json", True):
                result_path = output_directory / f"{_slug(dataset)}__{_slug(observable)}.json"
                result.to_json(result_path)
                output_files.extend(
                    [str(result_path), str(result_path.with_name(f"{result_path.stem}.manifest.json"))]
                )
            if outputs.get("plots", False):
                plot_path = output_directory / f"{_slug(dataset)}__{_slug(observable)}.png"
                figure, _ = result.plot(output_path=plot_path)
                import matplotlib.pyplot as plt

                plt.close(figure)
                output_files.append(str(plot_path))

    summary_path: Path | None = None
    summary_name = outputs.get("summary_csv", "summary.csv")
    if summary_name:
        summary_path = output_directory / str(summary_name)
        pd.DataFrame(
            [
                {
                    "dataset": result.dataset_name,
                    "observable": result.observable,
                    "method": result.method,
                    "model": result.model,
                    "baseline": result.baseline,
                    "baseline_uncertainty": result.baseline_uncertainty,
                    "r_squared": result.r_squared,
                    "residual_sum_squares": result.residual_sum_squares,
                }
                for result in results.values()
            ]
        ).to_csv(summary_path, index=False)
        output_files.append(str(summary_path))

    manifest_path = output_directory / "study.manifest.json"
    output_files.append(str(manifest_path))
    from extrapolation import __version__

    manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "study": config["study"],
        "configuration": config,
        "configuration_path": str(config_path),
        "database_path": str(db.db_path),
        "runs": [result.manifest() for result in results.values()],
        "output_files": output_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return StudyResult(
        name=config["study"],
        results=results,
        output_directory=output_directory,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


__all__ = ["StudyResult", "run_study"]
