"""Headless-safe plotting functions that consume completed results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from extrapolation.result import FitResult


def _save(fig, output_path: str | Path | None) -> None:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")


def _transform(model: str, x: np.ndarray, x_max: float) -> np.ndarray:
    scaled = x / x_max
    if model == "exponential":
        return scaled
    if model == "sqrt_exponential":
        return np.sqrt(np.maximum(scaled, 0.0))
    if model == "power_law":
        return np.log(np.maximum(scaled, 1e-12))
    raise ValueError(f"Unknown fitted model {model!r}")


def _curve(result: FitResult, x: np.ndarray) -> np.ndarray:
    b = result.parameters["B"]
    amplitude = result.parameters["A"]
    transformed = _transform(result.model, x, float(result.selected_data[result.x_col].max()))
    return result.baseline + amplitude * np.exp(-b * transformed)


def plot_fit(result: FitResult, *, output_path: str | Path | None = None):
    """Plot selected observations, the fitted curve, baseline, and reference."""
    x = result.selected_data[result.x_col].to_numpy(dtype=float)
    y = result.selected_data[result.observable].to_numpy(dtype=float)
    x_plot = np.linspace(float(x.min()), float(x.max()) * 1.5, 300)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, color="black", label="Selected data", zorder=3)
    ax.plot(x_plot, _curve(result, x_plot), label=result.model.replace("_", " "))
    ax.axhline(result.baseline, color="tab:blue", linestyle="--", label="Baseline")
    if result.baseline_uncertainty is not None:
        ax.fill_between(
            x_plot,
            result.baseline - result.baseline_uncertainty,
            result.baseline + result.baseline_uncertainty,
            color="tab:blue",
            alpha=0.12,
            label="Baseline uncertainty",
        )
    if result.reference_value is not None:
        ax.axhline(result.reference_value, color="tab:red", linestyle=":", label="Reference")
        if result.reference_uncertainty is not None:
            ax.fill_between(
                x_plot,
                result.reference_value - result.reference_uncertainty,
                result.reference_value + result.reference_uncertainty,
                color="tab:red",
                alpha=0.1,
                label="Reference uncertainty",
            )
    ax.set(title=f"{result.dataset_name}: {result.observable}", xlabel=result.x_col, ylabel=result.observable)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)
    return fig, ax


def plot_log(result: FitResult, *, output_path: str | Path | None = None):
    """Plot the solver's logarithmically linearized relationship."""
    x = result.selected_data[result.x_col].to_numpy(dtype=float)
    y = result.selected_data[result.observable].to_numpy(dtype=float)
    increasing = bool(result.metadata.get("is_increasing", False))
    difference = result.baseline - y if increasing else y - result.baseline
    valid = difference > 0.0
    if valid.sum() < 2:
        raise ValueError("Fitted baseline leaves fewer than two valid logarithm points.")
    tx = _transform(result.model, x, float(x.max()))
    tx_valid = tx[valid]
    log_difference = np.log(difference[valid])
    line_x = np.linspace(float(tx_valid.min()), float(tx_valid.max()), 250)
    line_y = np.log(abs(result.parameters["A"])) - result.parameters["B"] * line_x
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tx_valid, log_difference, color="black", label="Selected data")
    ax.plot(line_x, line_y, color="tab:orange", label=f"Fit (R²={result.r_squared:.5f})")
    if len(tx_valid) > 3:
        from scipy.stats import t

        degrees_of_freedom = len(tx_valid) - 2
        log_residuals = log_difference - (
            np.log(abs(result.parameters["A"]))
            - result.parameters["B"] * tx_valid
        )
        residual_variance = float(
            np.sum(log_residuals**2) / degrees_of_freedom
        )
        centered_sum = float(np.sum((tx_valid - tx_valid.mean()) ** 2))
        if centered_sum > 0.0:
            critical = float(t.ppf(0.975, degrees_of_freedom))
            mean_variance = residual_variance * (
                1.0 / len(tx_valid)
                + (line_x - tx_valid.mean()) ** 2 / centered_sum
            )
            mean_error = critical * np.sqrt(mean_variance)
            prediction_error = critical * np.sqrt(
                mean_variance + residual_variance
            )
            ax.fill_between(
                line_x,
                line_y - prediction_error,
                line_y + prediction_error,
                color="tab:orange",
                alpha=0.08,
                label="95% prediction band",
            )
            ax.fill_between(
                line_x,
                line_y - mean_error,
                line_y + mean_error,
                color="tab:orange",
                alpha=0.18,
                label="95% confidence band",
            )
    ax.set(
        title=f"{result.observable}: linearized {result.model.replace('_', ' ')} fit",
        xlabel="transformed basis",
        ylabel="log distance from baseline",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)
    return fig, ax


def plot_profile(result: FitResult, *, output_path: str | Path | None = None):
    """Plot the already-computed baseline scan; fitting is never rerun."""
    values = np.asarray(result.scan_values, dtype=float)
    scores = np.asarray(result.scan_scores, dtype=float)
    valid = np.isfinite(values) & np.isfinite(scores)
    if not valid.any():
        raise ValueError("This result does not contain a finite baseline scan profile.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values[valid], scores[valid], color="tab:green")
    ax.axvline(result.baseline, color="black", linestyle="--", label="Selected baseline")
    ax.set(
        title=f"{result.observable}: baseline profile",
        xlabel="candidate baseline",
        ylabel="linearized R²",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)
    return fig, ax


__all__ = ["plot_fit", "plot_log", "plot_profile"]
