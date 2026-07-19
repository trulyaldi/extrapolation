"""Stable, serializable scientific fitting result."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_json_value(item) for item in list(value)]
    if isinstance(value, pd.DataFrame):
        return [_json_value(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass
class FitResult:
    """Portable output from a completed fit; no live solver is required."""

    dataset_name: str
    observable: str
    x_col: str
    method: str
    model: str
    baseline: float
    baseline_uncertainty: float | None
    confidence_interval: tuple[float, float] | None
    parameters: dict[str, float]
    r_squared: float
    residual_sum_squares: float
    residuals: np.ndarray
    scan_values: np.ndarray
    scan_scores: np.ndarray
    selected_data: pd.DataFrame
    predictions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_value: float | None = None
    reference_uncertainty: float | None = None

    def summary(self) -> str:
        """Return a concise, display-ready scientific summary."""
        uncertainty = (
            "not computed"
            if self.baseline_uncertainty is None
            else f"{self.baseline_uncertainty:.8g}"
        )
        return (
            f"{self.dataset_name} · {self.observable}\n"
            f"method={self.method}, model={self.model}, "
            f"points={len(self.selected_data)}\n"
            f"baseline={self.baseline:.15g}, uncertainty={uncertainty}, "
            f"R²(log)={self.r_squared:.8g}, SSR={self.residual_sum_squares:.8g}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary with stable public field names."""
        return _json_value(
            {
                "dataset_name": self.dataset_name,
                "observable": self.observable,
                "x_col": self.x_col,
                "method": self.method,
                "model": self.model,
                "baseline": self.baseline,
                "baseline_uncertainty": self.baseline_uncertainty,
                "confidence_interval": self.confidence_interval,
                "parameters": self.parameters,
                "r_squared": self.r_squared,
                "residual_sum_squares": self.residual_sum_squares,
                "residuals": self.residuals,
                "scan_values": self.scan_values,
                "scan_scores": self.scan_scores,
                "selected_data": self.selected_data,
                "predictions": self.predictions,
                "reference_value": self.reference_value,
                "reference_uncertainty": self.reference_uncertainty,
                "metadata": self.metadata,
            }
        )

    def manifest(self, output_files: list[str] | None = None) -> dict[str, Any]:
        """Build a small reproducibility manifest for an explicit export."""
        try:
            from extrapolation import __version__
        except ImportError:  # pragma: no cover - defensive for partial installs
            __version__ = "unknown"
        selection = self.metadata.get("selection", {})
        return _json_value(
            {
                "manifest_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "package_version": __version__,
                "database_schema_versions": self.metadata.get(
                    "database_schema_versions", []
                ),
                "dataset_name": self.dataset_name,
                "source_hash": self.metadata.get("source_hash"),
                "observable": self.observable,
                "independent_column": self.x_col,
                "selected_row_indices": selection.get("row_indices", []),
                "selected_basis_range": selection.get("basis_range"),
                "method": self.method,
                "model": self.model,
                "numerical_options": self.metadata.get("options", {}),
                "uncertainty_options": {
                    "compute_uq": self.metadata.get("options", {}).get("compute_uq")
                },
                "output_files": output_files or [],
            }
        )

    def to_json(self, path: str | Path | None = None, *, indent: int = 2):
        """Serialize to text, or write JSON plus a sibling run manifest."""
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is None:
            return text
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        manifest_path = destination.with_name(f"{destination.stem}.manifest.json")
        output_files = [str(destination), str(manifest_path)]
        manifest_path.write_text(
            json.dumps(self.manifest(output_files), indent=indent, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    def plot(self, *, output_path: str | Path | None = None):
        from extrapolation.plotting import plot_fit

        return plot_fit(self, output_path=output_path)

    def plot_log(self, *, output_path: str | Path | None = None):
        from extrapolation.plotting import plot_log

        return plot_log(self, output_path=output_path)

    def plot_profile(self, *, output_path: str | Path | None = None):
        from extrapolation.plotting import plot_profile

        return plot_profile(self, output_path=output_path)


__all__ = ["FitResult"]

