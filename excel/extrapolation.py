"""
Excel functions (xlwings UDFs) for extraplus.

Each function converts its Excel ranges, calls the extraplus Python API
(``extraplus.extrapolate()``, ``Result.plot()``, ``Result.plot_linearized()``)
and formats the result for the cell, so Excel and Python give the same results
by construction.  This module contains no fitting, validation or plotting
logic of its own.

Excel functions:

    EXTRAP1 / EXTRAP2 / EXTRAP3
        Return the extrapolated infinite-limit value C.

    UNCERTAINTY1 / UNCERTAINTY2 / UNCERTAINTY3
        Return the uncertainty of C in the original y units.

    DECAY_RATE1 / DECAY_RATE2 / DECAY_RATE3
        Return the fitted/fixed B parameter.

    EXTRAP1_V / EXTRAP2_V / EXTRAP3_V
        Spill [C, uncertainty, B] vertically.

    EXTRAP1_H / EXTRAP2_H / EXTRAP3_H
        Spill [C, uncertainty, B] horizontally.

    EXTRAP1_PLOT / EXTRAP2_PLOT / EXTRAP3_PLOT
        Insert the fit plot (Result.plot()) below the cell.

    EXTRAP1_PLOT_LOG / EXTRAP2_PLOT_LOG / EXTRAP3_PLOT_LOG
        Insert the linearized-space plot (Result.plot_linearized()) below the cell.

    EXTRAPLUS_VERSION
        Return the installed extraplus package version.

Model mapping:

    1 -> exponential
    2 -> sqrt_exponential
    3 -> power_law

Invalid input does not raise into Excel: the non-plot functions return the
API's ValueError message after "Error: " (for example "Error: At least three
paired numeric x/y observations are required.").

Installation: the ``extraplus`` package must be importable. The Windows
installer (excel/install.bat) installs the package, this module and the
``extraplus.xlam`` add-in into a per-user folder; nothing has to sit next to
the workbook. See excel/README.md.
"""

from __future__ import annotations

import warnings
from functools import lru_cache, wraps
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Import the installed ``extraplus`` package.
# ---------------------------------------------------------------------------

try:
    import extraplus
    from extraplus import Result, extrapolate
except ImportError as exc:  # pragma: no cover - installation/configuration error
    raise ImportError(
        "Could not import 'extraplus'. Install the extraplus package into this "
        "Python environment (on Windows, run excel/install.bat)."
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


# ---------------------------------------------------------------------------
# Calling the API from a cell
# ---------------------------------------------------------------------------


def _cells(values: Any) -> tuple:
    """The values of an Excel range (or a scalar) in order, as a hashable tuple."""
    return tuple(np.asarray(values, dtype=object).reshape(-1))


def _extrapolate_quietly(x, y, model_type, b_fixed, n_fit) -> Result:
    """extrapolate() without its RuntimeWarning, which has nowhere to go in Excel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return extrapolate(x, y, model=model_type, b_fixed=b_fixed, n_fit=n_fit)


@lru_cache(maxsize=256)
def _cached_extrapolate(x_cells: tuple, y_cells: tuple, model_type: str,
                        b_fixed: Any, n_fit: Any) -> Result:
    """Cache repeated Excel calls that use exactly the same cell values."""
    return _extrapolate_quietly(x_cells, y_cells, model_type, b_fixed, n_fit)


def _run_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
) -> Result:
    """Shared implementation used by every numeric xlwings UDF below."""
    key = (_cells(x_range), _cells(y_range), model_type, b_fixed, n_fit)
    try:
        hash(key)
    except TypeError:   # e.g. a range passed as b_fixed: nothing to cache on
        return _extrapolate_quietly(*key)
    return _cached_extrapolate(*key)


def _excel_errors(function):
    """
    Return a short "Error: ..." text to Excel instead of raising.

    Only the failure path changes: when the wrapped UDF succeeds, its value is
    returned unchanged.  ValueError messages come from the extraplus input
    checks and are shown as they are; any other exception also shows its type.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # never hand a traceback to an Excel cell
            return f"Error: {type(exc).__name__}: {exc}"

    return wrapper


# ---------------------------------------------------------------------------
# Excel plot helpers
# ---------------------------------------------------------------------------


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
) -> str:
    """Shared implementation for all six Excel plot UDFs."""
    result = _extrapolate_quietly(x_range, y_range, model_type, b_fixed, n_fit)
    fig = result.plot_linearized() if log_plot else result.plot()

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
@_excel_errors
def EXTRAP1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).value


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
@_excel_errors
def EXTRAP2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).value


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
@_excel_errors
def EXTRAP3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).value


# ---------------------------------------------------------------------------
# Uncertainty values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def UNCERTAINTY1(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def UNCERTAINTY2(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential asymptote uncertainty in original y units."""
    return _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def UNCERTAINTY3(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    ).uncertainty


# ---------------------------------------------------------------------------
# Fitted B values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def DECAY_RATE1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential B for the normalized variable x/x_max."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).decay_rate


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def DECAY_RATE2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential B for sqrt(x/x_max)."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).decay_rate


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def DECAY_RATE3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law B for (x/x_max)^(-B)."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).decay_rate


# ---------------------------------------------------------------------------
# Combined spill functions: C, uncertainty, B
# ---------------------------------------------------------------------------


def _vertical_fit_output(result: Result):
    """Return C, uncertainty, and B as an explicit 3 x 1 Excel array."""
    return np.asarray(
        [
            [result.value],
            [result.uncertainty],
            [result.decay_rate],
        ],
        dtype=float,
    )


def _horizontal_fit_output(result: Result):
    """Return C, uncertainty, and B as an explicit 1 x 3 Excel array."""
    return np.asarray(
        [[result.value, result.uncertainty, result.decay_rate]],
        dtype=float,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP1_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP2_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP3_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP1_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP2_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def EXTRAP3_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


# ---------------------------------------------------------------------------
# Installed version
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@_excel_errors
def EXTRAPLUS_VERSION():
    """Installed version of the extraplus package."""
    return extraplus.__version__


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
    "EXTRAPLUS_VERSION",
    "EXTRAP1_PLOT",
    "EXTRAP2_PLOT",
    "EXTRAP3_PLOT",
    "EXTRAP1_PLOT_LOG",
    "EXTRAP2_PLOT_LOG",
    "EXTRAP3_PLOT_LOG",
]