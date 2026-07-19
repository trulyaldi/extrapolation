"""Canonical high-level fitting orchestration."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import io
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from database import DatasetDatabase
from extrapolation.data import FitValidationError, load_dataset_bundle, validate_fit_request
from extrapolation.fitting import VarProLinearized
from extrapolation.result import FitResult


SUPPORTED_MODELS = ("exponential", "sqrt_exponential", "power_law")
MODEL_ALIASES = {
    "exp": "exponential",
    "sqrt-exp": "sqrt_exponential",
    "sqrt_exponential": "sqrt_exponential",
    "power": "power_law",
    "power-law": "power_law",
}


def _metadata_default(metadata: dict[str, Any], key: str, fallback: Any) -> Any:
    defaults = metadata.get("fit_defaults")
    if isinstance(defaults, dict) and key in defaults:
        return defaults[key]
    return metadata.get(f"default_{key}", metadata.get(key, fallback))


def _model_name(model: str) -> str:
    normalized = model.casefold().replace(" ", "_")
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized != "auto" and normalized not in SUPPORTED_MODELS:
        supported = ", ".join(("auto", *SUPPORTED_MODELS))
        raise FitValidationError(f"Unsupported model {model!r}; choose one of {supported}.")
    return normalized


def _schema_versions(db: DatasetDatabase) -> list[str]:
    try:
        return list(db.schema_versions())
    except (RuntimeError, ValueError):
        return []


def fit_dataset(
    dataset: str,
    observable: str,
    *,
    db: DatasetDatabase | None = None,
    method: str | None = None,
    model: str = "auto",
    n_fit: int | None = None,
    basis_min: float | None = None,
    basis_max: float | None = None,
    use_energy_b: bool | None = None,
    compute_uq: bool = True,
    missing: str = "drop",
    verbose: bool = False,
    x_col: str | None = None,
    error_dataset: str | None = "auto",
    reference_dataset: str | None = "auto",
) -> FitResult:
    """Load, validate, select, and fit one catalogued observable.

    Rows are stably sorted by the independent variable.  ``n_fit`` then selects
    the highest-basis rows.  The existing :class:`VarProLinearized` engine does
    all numerical fitting; this function only orchestrates it.
    """

    db = db or DatasetDatabase()
    bundle = load_dataset_bundle(
        db,
        dataset,
        error_dataset=error_dataset,
        reference_dataset=reference_dataset,
    )
    metadata = bundle.metadata
    effective_method = str(
        method if method is not None else _metadata_default(metadata, "method", "linearized")
    ).casefold()
    if effective_method != "linearized":
        raise FitValidationError(
            f"Unsupported fitting method {effective_method!r}; only 'linearized' is available."
        )
    effective_model = _model_name(model)
    effective_n_fit = (
        n_fit if n_fit is not None else _metadata_default(metadata, "n_fit", None)
    )
    effective_basis_min = (
        basis_min
        if basis_min is not None
        else _metadata_default(metadata, "basis_min", None)
    )
    effective_use_energy_b = (
        bool(use_energy_b)
        if use_energy_b is not None
        else bool(_metadata_default(metadata, "use_energy_b", True))
    )
    effective_x_col = x_col or str(
        _metadata_default(metadata, "independent_column", bundle.x_col)
    )

    selection = validate_fit_request(
        bundle,
        observable,
        x_col=effective_x_col,
        basis_min=effective_basis_min,
        basis_max=basis_max,
        n_fit=effective_n_fit,
        missing=missing,
        use_energy_b=effective_use_energy_b,
    )
    models = list(SUPPORTED_MODELS) if effective_model == "auto" else [effective_model]
    quiet = redirect_stdout(io.StringIO()) if not verbose else nullcontext()
    try:
        with quiet:
            solver = VarProLinearized(
                selection.data,
                selection.x_col,
                observable,
                err_df=bundle.errors,
                inf_df=bundle.reference,
                n_fit=None,
                use_energy_b=effective_use_energy_b,
            )
            raw_results = solver.fit_linearized(
                models=models,
                verbose=verbose,
                compute_uq=compute_uq,
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise FitValidationError(
            f"Cannot fit observable {observable!r} in {bundle.dataset_name!r}: {error}"
        ) from error
    viable = {
        name: value
        for name, value in raw_results.items()
        if np.isfinite(value.get("r2_linearized", np.nan))
        and np.isfinite(value.get("C", np.nan))
    }
    if not viable:
        raise FitValidationError(
            f"Cannot fit observable {observable!r} in {bundle.dataset_name!r}: "
            "the baseline scan found no feasible logarithmic domain."
        )
    selected_model = (
        max(viable, key=lambda name: viable[name]["r2_linearized"])
        if effective_model == "auto"
        else effective_model
    )
    raw = viable[selected_model]

    y_range = float(solver.y_range)
    y_min = float(solver.y_min)
    baseline = y_min + y_range * float(raw["C"])
    amplitude = y_range * float(raw["A"])
    predictions = y_min + y_range * np.asarray(raw["y_pred"], dtype=float)
    observed = selection.data[observable].to_numpy(dtype=float)
    residuals = observed - predictions
    sigma_scaled = raw.get("sigma_mc") if compute_uq else None
    baseline_uncertainty = (
        y_range * float(sigma_scaled) if sigma_scaled is not None else None
    )
    if compute_uq and "sigma_C_plus" in raw and "sigma_C_minus" in raw:
        confidence_interval = (
            baseline - y_range * float(raw["sigma_C_minus"]),
            baseline + y_range * float(raw["sigma_C_plus"]),
        )
    elif baseline_uncertainty is not None:
        confidence_interval = (
            baseline - baseline_uncertainty,
            baseline + baseline_uncertainty,
        )
    else:
        confidence_interval = None

    scan_scaled, scan_scores = solver.scan_profiles.get(
        selected_model, (np.array([], dtype=float), np.array([], dtype=float))
    )
    scan_values = y_min + y_range * np.asarray(scan_scaled, dtype=float)
    reference_value = None
    if bundle.reference is not None and observable in bundle.reference:
        finite_reference = pd.to_numeric(
            bundle.reference[observable], errors="coerce"
        ).dropna()
        if not finite_reference.empty:
            reference_value = float(finite_reference.iloc[-1])
    reference_uncertainty = None
    if bundle.errors is not None and observable in bundle.errors:
        finite_error = pd.to_numeric(bundle.errors[observable], errors="coerce").dropna()
        if not finite_error.empty:
            reference_uncertainty = float(finite_error.iloc[-1])

    x_values = selection.data[selection.x_col].to_numpy(dtype=float)
    tau = kendalltau(x_values, observed).statistic
    result_metadata = {
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "dataset_metadata": metadata,
        "database_path": str(db.db_path),
        "database_schema_versions": _schema_versions(db),
        "relationships": {
            "error_dataset": bundle.error_dataset,
            "reference_dataset": bundle.reference_dataset,
            "resolution": bundle.relationship_sources,
        },
        "selection": {
            "row_indices": list(selection.original_row_indices),
            "dropped_missing_row_indices": list(selection.dropped_missing_rows),
            "basis_range": [float(x_values.min()), float(x_values.max())],
            "basis_min": effective_basis_min,
            "basis_max": basis_max,
            "n_fit": effective_n_fit,
            "sort": f"{selection.x_col} ascending; highest n_fit rows",
            "missing": missing,
        },
        "options": {
            "method": effective_method,
            "requested_model": effective_model,
            "compute_uq": compute_uq,
            "use_energy_b": effective_use_energy_b,
        },
        "candidate_scores": {
            name: float(value["r2_linearized"]) for name, value in viable.items()
        },
        "is_increasing": bool(tau > 0),
    }
    return FitResult(
        dataset_name=bundle.dataset_name,
        observable=observable,
        x_col=selection.x_col,
        method=effective_method,
        model=selected_model,
        baseline=baseline,
        baseline_uncertainty=baseline_uncertainty,
        confidence_interval=confidence_interval,
        parameters={"A": amplitude, "B": float(raw["B"]), "C": baseline},
        r_squared=float(raw["r2_linearized"]),
        residual_sum_squares=float(raw["ssr"]) * y_range**2,
        residuals=residuals,
        scan_values=scan_values,
        scan_scores=np.asarray(scan_scores, dtype=float),
        selected_data=selection.data.copy(),
        predictions=predictions,
        metadata=result_metadata,
        reference_value=reference_value,
        reference_uncertainty=reference_uncertainty,
    )


def fit_all_observables(
    dataset: str,
    *,
    observables: Iterable[str] | None = None,
    db: DatasetDatabase | None = None,
    on_error: str = "raise",
    **fit_options: Any,
) -> dict[str, FitResult]:
    """Fit selected/all numeric observables by delegating to :func:`fit_dataset`."""
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    db = db or DatasetDatabase()
    metadata = db.get_dataset_metadata(dataset.casefold())
    frame = db.load_observations(dataset.casefold())
    x_col = str(metadata["independent_column"])
    ignored = set(_metadata_default(metadata, "ignored_columns", ()))
    metadata_observables = _metadata_default(metadata, "observable_columns", None)
    if observables is not None:
        requested = list(observables)
    elif metadata_observables is not None:
        requested = list(metadata_observables)
    else:
        requested = [
            column
            for column in frame.columns
            if column != x_col and column not in ignored
        ]
    results: dict[str, FitResult] = {}
    failures: dict[str, str] = {}
    for observable in requested:
        try:
            results[observable] = fit_dataset(
                dataset,
                observable,
                db=db,
                **fit_options,
            )
        except (FitValidationError, KeyError, ValueError) as error:
            if on_error == "raise":
                raise
            failures[observable] = str(error)
    if failures:
        for result in results.values():
            result.metadata["batch_failures"] = failures
    return results


__all__ = ["SUPPORTED_MODELS", "fit_all_observables", "fit_dataset"]
