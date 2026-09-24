"""
Input checks shared by the Python API and the Excel functions.

The Excel functions show these messages in a cell as "Error: <message>", and
the Python API raises them as ``ValueError(<message>)``, so both read the same.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np

MODELS = ("exponential", "sqrt_exponential", "power_law")


def model_name(model: Any) -> str:
    """Return the model name for ``model`` given as a name or as 1, 2 or 3."""
    if isinstance(model, str) and model in MODELS:
        return model
    if isinstance(model, numbers.Integral) and not isinstance(model, bool) and 1 <= model <= 3:
        return MODELS[int(model) - 1]
    raise ValueError("model must be 'exponential', 'sqrt_exponential', 'power_law', 1, 2 or 3.")


def flatten(values: Any) -> np.ndarray:
    """Flatten a scalar, sequence, Series, row range or column range without reordering it."""
    if isinstance(values, np.ndarray):
        return values.astype(object, copy=False).reshape(-1)
    return np.asarray(values, dtype=object).reshape(-1)


def clean_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair x and y and keep the pairs where both are finite numbers.

    Blank cells (None or ""), text that is not a number, and non-finite values
    drop the whole pair.  Raises ValueError when what remains cannot be fitted.
    """
    x_raw = flatten(x)
    y_raw = flatten(y)

    if x_raw.size != y_raw.size:
        raise ValueError("x and y must have the same length.")

    x_values: list[float] = []
    y_values: list[float] = []

    for x_item, y_item in zip(x_raw, y_raw):
        if x_item is None or y_item is None or x_item == "" or y_item == "":
            continue

        try:
            x_value = float(x_item)
            y_value = float(y_item)
        except (TypeError, ValueError):
            # Headers, text and Excel error values are skipped.
            continue

        if np.isfinite(x_value) and np.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)

    if len(x_values) < 3:
        raise ValueError("At least three paired numeric x/y observations are required.")

    x_clean = np.asarray(x_values, dtype=float)
    y_clean = np.asarray(y_values, dtype=float)

    if np.unique(x_clean).size < 2:
        raise ValueError("x must contain at least two distinct values.")

    if float(np.max(x_clean)) == 0.0:
        raise ValueError("The largest x value cannot be zero because x is scaled by x_max.")

    if float(np.max(y_clean) - np.min(y_clean)) == 0.0:
        raise ValueError("y is constant, so an extrapolation cannot be fitted.")

    return x_clean, y_clean


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None


def optional_b_fixed(value: Any) -> Optional[float]:
    """None (fit B) or a finite, positive fixed B.  A blank Excel cell ("") means None."""
    if value is None or value == "":
        return None

    b_fixed = _number(value, "b_fixed")
    if not math.isfinite(b_fixed):
        raise ValueError("b_fixed must be finite.")
    if b_fixed <= 0.0:
        raise ValueError("b_fixed must be greater than zero.")
    return b_fixed


def optional_n_fit(value: Any, n_available: int) -> Optional[int]:
    """
    None (use every point) or the number of leading points to fit: truncated
    to a whole number, at least 3, and capped at ``n_available``.  A blank
    Excel cell ("") means None.
    """
    if value is None or value == "":
        return None

    number = _number(value, "n_fit")
    if not math.isfinite(number):
        raise ValueError("n_fit must be finite.")
    n_fit = int(number)
    if n_fit < 3:
        raise ValueError("n_fit must be at least 3.")
    return min(n_fit, n_available)


def _target(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("target is required.")

    target = _number(value, "target")
    if not math.isfinite(target):
        raise ValueError("target must be finite.")
    if target <= 0.0:
        raise ValueError("target must be greater than zero.")
    return target


def targets(value: Any) -> np.ndarray:
    """
    The basis sizes to predict at, as a float array of the input's shape (0-d
    for a scalar).  Each must be a finite number greater than zero.
    """
    items = np.asarray(value, dtype=object)
    return np.array([_target(item) for item in items.reshape(-1)], dtype=float).reshape(items.shape)
