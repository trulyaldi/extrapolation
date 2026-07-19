"""Dataset relationship resolution and pre-fit data validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from database import DatasetDatabase


class FitValidationError(ValueError):
    """Raised when a fit request cannot safely enter the numerical solver."""


@dataclass(frozen=True)
class DatasetBundle:
    """Observation data and its optional uncertainty/reference companions."""

    dataset_name: str
    observations: pd.DataFrame
    errors: pd.DataFrame | None
    reference: pd.DataFrame | None
    metadata: dict[str, Any]
    x_col: str
    error_dataset: str | None = None
    reference_dataset: str | None = None
    relationship_sources: dict[str, str] | None = None


@dataclass(frozen=True)
class FitSelection:
    """Validated, deterministically ordered input to the solver."""

    data: pd.DataFrame
    original_row_indices: tuple[int, ...]
    dropped_missing_rows: tuple[int, ...]
    x_col: str


def _metadata_relationship(metadata: dict[str, Any], kind: str) -> str | None:
    relationships = metadata.get("relationships")
    if isinstance(relationships, dict):
        value = relationships.get(kind)
        if value:
            return str(value)
    for key in (f"related_{kind}_dataset", f"{kind}_dataset"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _fallback_relationship(dataset_name: str, kind: str) -> str:
    if "/" not in dataset_name:
        prefix, stem = dataset_name, dataset_name
    else:
        prefix, stem = dataset_name.split("/", 1)
    if kind == "error":
        return f"{prefix}-err/{stem}_err".casefold()
    return f"{prefix}-init/{stem}".casefold()


def _resolve_relationship(
    *,
    dataset_name: str,
    kind: Literal["error", "reference"],
    requested: str | None | Literal["auto"],
    metadata: dict[str, Any],
    available: set[str],
) -> tuple[str | None, str]:
    if requested is None:
        return None, "disabled"
    if requested != "auto":
        explicit = requested.casefold()
        if explicit not in available:
            raise FitValidationError(
                f"Explicit {kind} dataset {requested!r} does not exist."
            )
        return explicit, "argument"

    configured = _metadata_relationship(metadata, kind)
    if configured:
        configured = configured.casefold()
        if configured not in available:
            raise FitValidationError(
                f"Metadata for {dataset_name!r} names missing {kind} dataset "
                f"{configured!r}."
            )
        return configured, "metadata"

    candidate = _fallback_relationship(dataset_name, kind)
    if candidate in available:
        return candidate, "naming convention"
    return None, "not found"


def load_dataset_bundle(
    db: DatasetDatabase,
    dataset_name: str,
    *,
    error_dataset: str | None | Literal["auto"] = "auto",
    reference_dataset: str | None | Literal["auto"] = "auto",
) -> DatasetBundle:
    """Load observations and resolve optional companions deterministically.

    Explicit arguments win, followed by relationship metadata and then the
    ``<directory>-err/<stem>_err`` / ``<directory>-init/<stem>`` convention.
    Passing ``None`` explicitly disables either relationship.
    """

    logical_name = dataset_name.casefold()
    try:
        metadata = db.get_dataset_metadata(logical_name)
        observations = db.load_observations(logical_name)
    except KeyError as error:
        raise FitValidationError(f"Dataset {dataset_name!r} does not exist.") from error

    available = set(db.list_datasets()["dataset_name"].astype(str))
    error_name, error_source = _resolve_relationship(
        dataset_name=logical_name,
        kind="error",
        requested=error_dataset,
        metadata=metadata,
        available=available,
    )
    reference_name, reference_source = _resolve_relationship(
        dataset_name=logical_name,
        kind="reference",
        requested=reference_dataset,
        metadata=metadata,
        available=available,
    )
    if error_name:
        error_metadata = db.get_dataset_metadata(error_name)
        if error_metadata.get("data_role") != "uncertainty":
            raise FitValidationError(
                f"Related error dataset {error_name!r} has role "
                f"{error_metadata.get('data_role')!r}, not 'uncertainty'."
            )
        errors = db.load_dataset(error_name)
    else:
        errors = None
    reference = db.load_dataset(reference_name) if reference_name else None
    return DatasetBundle(
        dataset_name=logical_name,
        observations=observations,
        errors=errors,
        reference=reference,
        metadata=metadata,
        x_col=str(metadata["independent_column"]),
        error_dataset=error_name,
        reference_dataset=reference_name,
        relationship_sources={
            "error": error_source,
            "reference": reference_source,
        },
    )


def validate_fit_request(
    bundle: DatasetBundle,
    observable: str,
    *,
    x_col: str | None = None,
    basis_min: float | None = None,
    basis_max: float | None = None,
    n_fit: int | None = None,
    missing: str = "drop",
    use_energy_b: bool = True,
) -> FitSelection:
    """Validate and select rows, sorting by basis and taking the highest ``n_fit``."""

    prefix = f"Cannot fit observable {observable!r} in {bundle.dataset_name!r}:"
    x_col = x_col or bundle.x_col
    if missing not in {"drop", "raise"}:
        raise FitValidationError(f"{prefix} missing must be 'drop' or 'raise'.")
    if x_col not in bundle.observations.columns:
        raise FitValidationError(f"{prefix} independent column {x_col!r} is absent.")
    if observable not in bundle.observations.columns:
        raise FitValidationError(f"{prefix} the observable column is absent.")
    if n_fit is not None and (
        isinstance(n_fit, bool) or not isinstance(n_fit, (int, np.integer)) or n_fit < 3
    ):
        raise FitValidationError(f"{prefix} n_fit must be an integer of at least 3.")
    for name, bound in (("basis_min", basis_min), ("basis_max", basis_max)):
        if bound is not None and not np.isfinite(float(bound)):
            raise FitValidationError(f"{prefix} {name} must be finite.")
    if basis_min is not None and basis_max is not None and basis_min > basis_max:
        raise FitValidationError(f"{prefix} basis_min cannot exceed basis_max.")

    frame = bundle.observations.copy()
    required = [x_col, observable]
    if use_energy_b and observable != "Energy":
        if "Energy" not in frame.columns:
            raise FitValidationError(
                f"{prefix} Energy is required when use_energy_b=True."
            )
        required.append("Energy")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["__source_row__"] = np.arange(len(frame), dtype=int)
    finite_mask = np.ones(len(frame), dtype=bool)
    for column in required:
        finite_mask &= np.isfinite(frame[column].to_numpy(dtype=float))
    dropped = tuple(frame.loc[~finite_mask, "__source_row__"].astype(int))
    if dropped and missing == "raise":
        raise FitValidationError(
            f"{prefix} rows {list(dropped)} contain missing or non-finite required values."
        )
    frame = frame.loc[finite_mask]
    if basis_min is not None:
        frame = frame.loc[frame[x_col] >= float(basis_min)]
    if basis_max is not None:
        frame = frame.loc[frame[x_col] <= float(basis_max)]
    frame = frame.sort_values(x_col, kind="mergesort")
    if n_fit is not None:
        frame = frame.tail(int(n_fit))

    if len(frame) < 3:
        raise FitValidationError(
            f"{prefix} only {len(frame)} finite points remain after filtering; "
            "at least 3 are required."
        )
    duplicates = frame[x_col].duplicated(keep=False)
    if duplicates.any():
        values = sorted(set(frame.loc[duplicates, x_col].astype(float)))
        raise FitValidationError(
            f"{prefix} duplicate independent-variable values remain: {values}."
        )
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[observable].to_numpy(dtype=float)
    if np.any(x <= 0.0):
        raise FitValidationError(
            f"{prefix} independent-variable values must be positive for the supported models."
        )
    if np.ptp(x) <= 0.0:
        raise FitValidationError(f"{prefix} independent-variable values have zero span.")
    if np.ptp(y) <= 0.0:
        raise FitValidationError(f"{prefix} observable values have zero span.")

    if bundle.errors is not None:
        if observable not in bundle.errors.columns:
            raise FitValidationError(
                f"{prefix} related error data lacks the observable column."
            )
        error_values = pd.to_numeric(bundle.errors[observable], errors="coerce")
        error_array = error_values.to_numpy(dtype=float)
        if (
            error_array.size == 0
            or not np.isfinite(error_array).all()
            or (error_array <= 0).any()
        ):
            raise FitValidationError(
                f"{prefix} related error values must be finite and positive."
            )
        if len(bundle.errors) not in {1, len(bundle.observations)}:
            raise FitValidationError(
                f"{prefix} related errors must contain one terminal estimate or one "
                "row per observation."
            )
        if len(bundle.errors) == len(bundle.observations):
            if x_col not in bundle.errors.columns:
                raise FitValidationError(
                    f"{prefix} row-aligned errors lack independent column {x_col!r}."
                )
            error_x = pd.to_numeric(bundle.errors[x_col], errors="coerce").to_numpy(
                dtype=float
            )
            observation_x = pd.to_numeric(
                bundle.observations[x_col], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(error_x).all() or not np.array_equal(
                error_x, observation_x
            ):
                raise FitValidationError(
                    f"{prefix} observation and error independent-variable rows do not align."
                )

    if bundle.reference is not None:
        if observable not in bundle.reference.columns:
            raise FitValidationError(
                f"{prefix} related reference data lacks the observable column."
            )
        reference_values = pd.to_numeric(bundle.reference[observable], errors="coerce")
        if not np.isfinite(reference_values.to_numpy(dtype=float)).any():
            raise FitValidationError(
                f"{prefix} related reference data has no finite value."
            )

    original_rows = tuple(frame.pop("__source_row__").astype(int))
    return FitSelection(
        data=frame.reset_index(drop=True),
        original_row_indices=original_rows,
        dropped_missing_rows=dropped,
        x_col=x_col,
    )


__all__ = [
    "DatasetBundle",
    "FitSelection",
    "FitValidationError",
    "load_dataset_bundle",
    "validate_fit_request",
]
