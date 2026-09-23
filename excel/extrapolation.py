"""
Excel-callable wrappers around ``extrap.py``'s ``extraplus`` class.

This module intentionally contains NO extrapolation algorithm of its own.
All fitting and uncertainty calculations are delegated to ``extrap.extraplus``.

The Excel-facing API is kept compatible with the previous extrapolation.py:

    EXTRAP1 / EXTRAP2 / EXTRAP3
        Return the extrapolated infinite-limit value C.

    UNCERTY1 / UNCERTY2 / UNCERTY3
        Return the uncertainty of C in the original y units.

    PARAMETER_B1 / PARAMETER_B2 / PARAMETER_B3
        Return the fitted/fixed B parameter.

    EXTRAP1_V / EXTRAP2_V / EXTRAP3_V
        Spill [C, uncertainty, B] vertically.

    EXTRAP1_H / EXTRAP2_H / EXTRAP3_H
        Spill [C, uncertainty, B] horizontally.

    EXTRAP1_PLOT / EXTRAP2_PLOT / EXTRAP3_PLOT
        Insert extrap.py's standard Matplotlib fit plot into Excel.

    EXTRAP1_PLOT_LOG / EXTRAP2_PLOT_LOG / EXTRAP3_PLOT_LOG
        Insert extrap.py's logarithmic Matplotlib plot into Excel.

Model mapping:

    1 -> exponential
    2 -> sqrt_exponential
    3 -> power_law

Expected folder layout:

    your_folder/
        extrap.py
        extrapolation.py
        your_workbook.xlsm

In Excel, set xlwings ``UDF Modules`` to ``extrapolation`` and import the
Python UDFs as before.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Import extraplus from extrap.py next to THIS file.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from extrap import extraplus
except ImportError as exc:  # pragma: no cover - installation/configuration error
    raise ImportError(
        "Could not import 'extraplus' from extrap.py. "
        "Place extrap.py in the same folder as extrapolation.py."
    ) from exc


# xlwings is optional for normal Python testing, but required inside Excel.
try:
    import xlwings as xw
except ImportError:  # pragma: no cover
    class _XlwingsStub:
        def func(self, function=None, **_kwargs):
            if function is None:
                return lambda fn: fn
            return function

        def arg(self, *_args, **_kwargs):
            return lambda fn: fn

        def ret(self, *_args, **_kwargs):
            return lambda fn: fn

    xw = _XlwingsStub()  # type: ignore[assignment]


_EXPONENTIAL = "exponential"
_SQRT_EXPONENTIAL = "sqrt_exponential"
_POWER_LAW = "power_law"

_X_COL = "__x__"
_Y_COL = "__y__"


@dataclass(frozen=True)
class _FitOutput:
    """Small immutable result object used by the Excel wrappers."""

    extrapolated: float
    uncertainty: float
    B: float
    A: float
    r2_log: float
    ssr_scaled: float


# ---------------------------------------------------------------------------
# Excel input cleaning
# ---------------------------------------------------------------------------


def _flatten_excel_range(values: Any) -> np.ndarray:
    """Flatten a scalar, row range, or column range without reordering it."""
    if isinstance(values, np.ndarray):
        return values.astype(object, copy=False).reshape(-1)
    return np.asarray(values, dtype=object).reshape(-1)


def _clean_xy(x_range: Any, y_range: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert Excel ranges to paired finite float arrays, dropping blank/text rows."""
    x_raw = _flatten_excel_range(x_range)
    y_raw = _flatten_excel_range(y_range)

    if x_raw.size != y_raw.size:
        raise ValueError("x_range and y_range must contain the same number of cells.")

    x_values: list[float] = []
    y_values: list[float] = []

    for x_item, y_item in zip(x_raw, y_raw):
        if x_item is None or y_item is None or x_item == "" or y_item == "":
            continue

        try:
            x_value = float(x_item)
            y_value = float(y_item)
        except (TypeError, ValueError):
            # Headers, text, and Excel error strings are ignored exactly as before.
            continue

        if np.isfinite(x_value) and np.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)

    if len(x_values) < 3:
        raise ValueError("At least three paired numeric x/y observations are required.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if np.unique(x).size < 2:
        raise ValueError("x_range must contain at least two distinct numeric values.")

    if float(np.max(x)) == 0.0:
        raise ValueError("The largest x value cannot be zero because x is scaled by x_max.")

    if float(np.max(y) - np.min(y)) == 0.0:
        raise ValueError("y_range is constant, so an extrapolation cannot be fitted.")

    return x, y


def _optional_float(value: Any) -> Optional[float]:
    """Convert the optional Excel B argument to a validated float."""
    if value is None or value == "":
        return None

    result = float(value)
    if not np.isfinite(result):
        raise ValueError("b_fixed must be finite.")
    if result <= 0.0:
        raise ValueError("b_fixed must be greater than zero.")
    return result


def _optional_n_fit(value: Any, n_available: int) -> Optional[int]:
    """Convert the optional Excel n_fit argument to an integer."""
    if value is None or value == "":
        return None

    n_fit = int(float(value))
    if n_fit < 3:
        raise ValueError("n_fit must be at least 3.")
    return min(n_fit, n_available)


def _as_bool(value: Any, default: bool) -> bool:
    """Handle Excel TRUE/FALSE values as well as common string forms."""
    if value is None or value == "":
        return default

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False

    return bool(value)


# ---------------------------------------------------------------------------
# Thin adapter around extrap.extraplus
# ---------------------------------------------------------------------------


def _fit_with_extraplus(
    x_values: np.ndarray,
    y_values: np.ndarray,
    model_type: str,
    fixed_b: Optional[float],
    n_fit: Optional[int],
    k_cov: float,
    q: float,
    use_pi: bool,
    robust_scale: bool,
) -> _FitOutput:
    """
    Run one model through ``extraplus`` and convert its scaled outputs back to
    the units expected by the existing Excel functions.

    ``use_energy_b=False`` is deliberate: the Excel wrapper already implements
    Energy-B propagation explicitly through the ``b_fixed`` cell argument.
    """
    df = pd.DataFrame({_X_COL: x_values, _Y_COL: y_values})

    # extraplus currently prints the trend direction in __init__.  Suppress that
    # console output because Excel UDF calls should remain silent.
    with redirect_stdout(StringIO()):
        solver = extraplus(
            df=df,
            x_col=_X_COL,
            y_col=_Y_COL,
            err_df=None,
            inf_df=None,
            b_init=fixed_b,
            n_fit=n_fit,
            use_energy_b=False,
        )

        # Fit first without UQ so the Excel UQ options below are applied exactly.
        results = solver.fit_linearized(
            models=[model_type],
            verbose=False,
            compute_uq=False,
        )

        if model_type not in results:
            raise ValueError(f"extraplus did not return a result for '{model_type}'.")

        solver.compute_uncertainty(
            k_cov=k_cov,
            q=q,
            use_PI=use_pi,
            robust_scale=robust_scale,
        )

    result = solver.results[model_type]

    C_scaled = float(result["C"])
    A_scaled = float(result["A"])
    B = float(result["B"])
    r2_log = float(result["r2_linearized"])
    ssr_scaled = float(result["ssr"])

    if "sigma_C" in result:
        sigma_C_scaled = float(result["sigma_C"])
    elif "sigma_mc" in result:
        # Compatibility with older extrap.py versions.
        sigma_C_scaled = float(result["sigma_mc"])
    else:
        raise ValueError("extraplus did not produce sigma_C uncertainty.")

    extrapolated = solver.y_min + solver.y_range * C_scaled
    uncertainty = solver.y_range * sigma_C_scaled
    A_unscaled = solver.y_range * A_scaled

    return _FitOutput(
        extrapolated=float(extrapolated),
        uncertainty=float(uncertainty),
        B=B,
        A=float(A_unscaled),
        r2_log=r2_log,
        ssr_scaled=ssr_scaled,
    )


@lru_cache(maxsize=256)
def _cached_fit(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    model_type: str,
    fixed_b: Optional[float],
    n_fit: Optional[int],
    k_cov: float,
    q: float,
    use_pi: bool,
    robust_scale: bool,
) -> _FitOutput:
    """Cache repeated Excel calls that use exactly the same inputs."""
    return _fit_with_extraplus(
        x_values=np.asarray(x_values, dtype=float),
        y_values=np.asarray(y_values, dtype=float),
        model_type=model_type,
        fixed_b=fixed_b,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


def _run_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
    k_cov: Any = 1.0,
    q: Any = 0.84,
    use_pi: Any = True,
    robust_scale: Any = True,
) -> _FitOutput:
    """Shared implementation used by every xlwings UDF below."""
    x, y = _clean_xy(x_range, y_range)

    fixed_b_value = _optional_float(b_fixed)
    n_fit_value = _optional_n_fit(n_fit, len(x))

    k_cov_value = float(k_cov if k_cov not in (None, "") else 1.0)
    q_value = float(q if q not in (None, "") else 0.84)

    if not np.isfinite(k_cov_value) or k_cov_value < 0.0:
        raise ValueError("k_cov must be a finite number greater than or equal to zero.")

    if not np.isfinite(q_value) or not 0.0 < q_value < 1.0:
        raise ValueError("q must be strictly between 0 and 1.")

    return _cached_fit(
        tuple(float(value) for value in x),
        tuple(float(value) for value in y),
        model_type,
        fixed_b_value,
        n_fit_value,
        k_cov_value,
        q_value,
        _as_bool(use_pi, True),
        _as_bool(robust_scale, True),
    )



# ---------------------------------------------------------------------------
# Excel plot helpers
# ---------------------------------------------------------------------------


def _build_fitted_solver_for_plot(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
    k_cov: Any = 1.0,
    q: Any = 0.84,
    use_pi: Any = True,
    robust_scale: Any = True,
):
    """
    Build an ``extraplus`` instance for exactly one model.

    The fitting itself still lives entirely in extrap.py.  This helper only
    prepares Excel input, runs the requested extraplus model, and computes the
    uncertainty so that extrap.py's existing plot/plot_log methods have the
    same fitted result information available as the numerical Excel functions.
    """
    x, y = _clean_xy(x_range, y_range)

    fixed_b_value = _optional_float(b_fixed)
    n_fit_value = _optional_n_fit(n_fit, len(x))

    k_cov_value = float(k_cov if k_cov not in (None, "") else 1.0)
    q_value = float(q if q not in (None, "") else 0.84)

    if not np.isfinite(k_cov_value) or k_cov_value < 0.0:
        raise ValueError("k_cov must be a finite number greater than or equal to zero.")

    if not np.isfinite(q_value) or not 0.0 < q_value < 1.0:
        raise ValueError("q must be strictly between 0 and 1.")

    df = pd.DataFrame({_X_COL: x, _Y_COL: y})

    with redirect_stdout(StringIO()):
        solver = extraplus(
            df=df,
            x_col=_X_COL,
            y_col=_Y_COL,
            err_df=None,
            inf_df=None,
            b_init=fixed_b_value,
            n_fit=n_fit_value,
            use_energy_b=False,
        )

        results = solver.fit_linearized(
            models=[model_type],
            verbose=False,
            compute_uq=False,
        )

        if model_type not in results:
            raise ValueError(f"extraplus did not return a result for '{model_type}'.")

        solver.compute_uncertainty(
            k_cov=k_cov_value,
            q=q_value,
            use_PI=_as_bool(use_pi, True),
            robust_scale=_as_bool(robust_scale, True),
        )

    return solver


def _get_figure_from_extraplus(solver, log_plot: bool) -> Figure:
    """
    Call extrap.py's existing plot() or plot_log() method and recover the
    Matplotlib Figure that it created.

    This supports both styles of plotting method:
      * returning a Figure explicitly, or
      * creating a pyplot figure and returning None.

    plt.show() is temporarily suppressed because Excel needs the Figure object
    inserted into the worksheet rather than a separate GUI plot window.
    """
    plot_method = solver.plot_log if log_plot else solver.plot

    figures_before = set(plt.get_fignums())
    original_show = plt.show

    try:
        plt.show = lambda *args, **kwargs: None
        returned = plot_method()
    finally:
        plt.show = original_show

    if isinstance(returned, Figure):
        return returned

    # Also support plotting helpers that return a Matplotlib Axes.
    returned_figure = getattr(returned, "figure", None)
    if isinstance(returned_figure, Figure):
        return returned_figure

    figures_after = list(plt.get_fignums())
    new_figure_numbers = [
        number for number in figures_after if number not in figures_before
    ]

    if new_figure_numbers:
        return plt.figure(new_figure_numbers[-1])

    if figures_after:
        return plt.gcf()

    method_name = "plot_log()" if log_plot else "plot()"
    raise RuntimeError(
        f"extraplus.{method_name} did not return or create a Matplotlib Figure."
    )


def _insert_plot_in_excel(
    caller: Any,
    fig: Figure,
    picture_name: str,
) -> None:
    """
    Insert/update the Matplotlib figure directly below the Excel formula cell.

    The picture name includes the calling cell, so several EXTRAP*_PLOT
    formulas can coexist on the same worksheet.  update=True means that
    recalculation replaces the existing picture instead of creating duplicates.
    """
    if caller is None:
        raise RuntimeError(
            "This plot function must be called from an Excel worksheet through xlwings."
        )

    safe_name = (
        f"Extraplus_{picture_name}_R{int(caller.row)}_C{int(caller.column)}"
    )

    caller.sheet.pictures.add(
        fig,
        name=safe_name,
        update=True,
        left=caller.left,
        top=caller.top + caller.height,
    )


def _run_plot_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    caller: Any,
    picture_name: str,
    log_plot: bool,
    b_fixed: Any = None,
    n_fit: Any = None,
    k_cov: Any = 1.0,
    q: Any = 0.84,
    use_pi: Any = True,
    robust_scale: Any = True,
) -> str:
    """Shared implementation for all six Excel plot UDFs."""
    solver = _build_fitted_solver_for_plot(
        x_range=x_range,
        y_range=y_range,
        model_type=model_type,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )

    fig = _get_figure_from_extraplus(solver, log_plot=log_plot)

    try:
        _insert_plot_in_excel(
            caller=caller,
            fig=fig,
            picture_name=picture_name,
        )
    finally:
        plt.close(fig)

    return f"{picture_name} updated"


# ---------------------------------------------------------------------------
# Extrapolated values at infinity
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).extrapolated


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).extrapolated


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).extrapolated


# ---------------------------------------------------------------------------
# Uncertainty values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def UNCERTAINTY1(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Exponential asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def UNCERTAINTY2(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Square-root-exponential asymptote uncertainty in original y units."""
    return _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def UNCERTAINTY3(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Power-law asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    ).uncertainty


# ---------------------------------------------------------------------------
# Fitted B values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def DECAY_RATE1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential B for the normalized variable x/x_max."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).B


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def DECAY_RATE2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential B for sqrt(x/x_max)."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).B


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def DECAY_RATE3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law B for (x/x_max)^(-B)."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).B


# ---------------------------------------------------------------------------
# Combined spill functions: C, uncertainty, B
# ---------------------------------------------------------------------------


def _vertical_fit_output(result: _FitOutput):
    """Return C, uncertainty, and B as an explicit 3 x 1 Excel array."""
    return np.asarray(
        [
            [result.extrapolated],
            [result.uncertainty],
            [result.B],
        ],
        dtype=float,
    )


def _horizontal_fit_output(result: _FitOutput):
    """Return C, uncertainty, and B as an explicit 1 x 3 Excel array."""
    return np.asarray(
        [[result.extrapolated, result.uncertainty, result.B]],
        dtype=float,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP1_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP2_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP3_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Power-law fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP1_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP2_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
def EXTRAP3_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """Power-law fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
        k_cov,
        q,
        use_pi,
        robust_scale,
    )
    return _horizontal_fit_output(result)


# ---------------------------------------------------------------------------
# Matplotlib plots in Excel
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP1_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the exponential model and insert extrap.py's standard Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP1_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP2_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the square-root-exponential model and insert extrap.py's standard
    Matplotlib plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_SQRT_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP2_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP3_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the power-law model and insert extrap.py's standard Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_POWER_LAW,
        caller=caller,
        picture_name="EXTRAP3_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP1_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the exponential model and insert extrap.py's logarithmic Matplotlib
    plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP1_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP2_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the square-root-exponential model and insert extrap.py's logarithmic
    Matplotlib plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_SQRT_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP2_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP3_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
    k_cov=1.0,
    q=0.84,
    use_pi=True,
    robust_scale=True,
):
    """
    Fit the power-law model and insert extrap.py's logarithmic Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_POWER_LAW,
        caller=caller,
        picture_name="EXTRAP3_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
        k_cov=k_cov,
        q=q,
        use_pi=use_pi,
        robust_scale=robust_scale,
    )



__all__ = [
    "EXTRAP1",
    "EXTRAP2",
    "EXTRAP3",
    "UNCERTAINTY1",
    "UNCERTAINTY2",
    "UNCERTAINTY3",
    "DECAY_RATE1",
    "DECAY_RATE2",
    "DECAY_RATE3",
    "EXTRAP1_V",
    "EXTRAP2_V",
    "EXTRAP3_V",
    "EXTRAP1_H",
    "EXTRAP2_H",
    "EXTRAP3_H",
    "EXTRAP1_PLOT",
    "EXTRAP2_PLOT",
    "EXTRAP3_PLOT",
    "EXTRAP1_PLOT_LOG",
    "EXTRAP2_PLOT_LOG",
    "EXTRAP3_PLOT_LOG",
]