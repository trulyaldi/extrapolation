"""
Excel-callable wrappers around the ``extraplus`` class from the ``extraplus`` package.

This module intentionally contains NO extrapolation algorithm of its own.
All fitting and uncertainty calculations are delegated to ``extraplus.extraplus``.

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
        Insert the standard extraplus Matplotlib fit plot into Excel.

    EXTRAP1_PLOT_LOG / EXTRAP2_PLOT_LOG / EXTRAP3_PLOT_LOG
        Insert the logarithmic extraplus Matplotlib plot into Excel.

    EXTRAPLUS_VERSION
        Return the installed extraplus package version.

Model mapping:

    1 -> exponential
    2 -> sqrt_exponential
    3 -> power_law

Invalid input does not raise into Excel: the non-plot functions return a short
text starting with "Error: " (for example "Error: At least three paired numeric
x/y observations are required.").

Installation: the ``extraplus`` package must be importable. The Windows
installer (excel/install.bat) installs the package, this module and the
``extraplus.xlam`` add-in into a per-user folder; nothing has to sit next to
the workbook. See excel/README.md.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache, wraps
from importlib.metadata import version as _package_version
from io import StringIO
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


# ---------------------------------------------------------------------------
# Import extraplus from the installed ``extraplus`` package.
# ---------------------------------------------------------------------------

try:
    from extraplus import extraplus
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


# ---------------------------------------------------------------------------
# Thin adapter around extraplus.extraplus
# ---------------------------------------------------------------------------


def _fit_with_extraplus(
    x_values: np.ndarray,
    y_values: np.ndarray,
    model_type: str,
    fixed_b: Optional[float],
    n_fit: Optional[int],
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

        # Fit first without UQ, then compute the uncertainty in its own call.
        results = solver.fit_linearized(
            models=[model_type],
            verbose=False,
            compute_uq=False,
        )

        if model_type not in results:
            raise ValueError(f"extraplus did not return a result for '{model_type}'.")

        solver.compute_uncertainty()

    result = solver.results[model_type]

    C_scaled = float(result["C"])
    A_scaled = float(result["A"])
    B = float(result["B"])
    r2_log = float(result["r2_linearized"])
    ssr_scaled = float(result["ssr"])

    if "sigma_C" in result:
        sigma_C_scaled = float(result["sigma_C"])
    elif "sigma_mc" in result:
        # Compatibility with older extraplus versions.
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
) -> _FitOutput:
    """Cache repeated Excel calls that use exactly the same inputs."""
    return _fit_with_extraplus(
        x_values=np.asarray(x_values, dtype=float),
        y_values=np.asarray(y_values, dtype=float),
        model_type=model_type,
        fixed_b=fixed_b,
        n_fit=n_fit,
    )


def _run_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
) -> _FitOutput:
    """Shared implementation used by every xlwings UDF below."""
    x, y = _clean_xy(x_range, y_range)

    fixed_b_value = _optional_float(b_fixed)
    n_fit_value = _optional_n_fit(n_fit, len(x))

    return _cached_fit(
        tuple(float(value) for value in x),
        tuple(float(value) for value in y),
        model_type,
        fixed_b_value,
        n_fit_value,
    )


def _excel_errors(function):
    """
    Return a short "Error: ..." text to Excel instead of raising.

    Only the failure path changes: when the wrapped UDF succeeds, its value is
    returned unchanged.  ValueError messages come from the input checks above
    and are shown as they are; any other exception also shows its type.
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


def _build_fitted_solver_for_plot(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
):
    """
    Build an ``extraplus`` instance for exactly one model; return it with the
    cleaned x and y (all rows, including any beyond n_fit).

    The fitting itself still lives entirely in extrap.py.  This helper only
    prepares Excel input, runs the requested extraplus model, and computes the
    uncertainty so that extrap.py's existing plot/plot_log methods have the
    same fitted result information available as the numerical Excel functions.
    """
    x, y = _clean_xy(x_range, y_range)

    fixed_b_value = _optional_float(b_fixed)
    n_fit_value = _optional_n_fit(n_fit, len(x))

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

        solver.compute_uncertainty()

    return solver, x, y


def _figures_from_extraplus(solver, log_plot: bool) -> list[Figure]:
    """
    Call extrap.py's existing plot() or plot_log() method and return every
    Matplotlib Figure it created, oldest first.

    This supports plotting methods that return a Figure (or an Axes) as well as
    ones that only create pyplot figures and return None.  plt.show() is
    suppressed and the printed summary table is discarded, because Excel needs
    the Figure objects rather than a GUI window or console output.
    """
    plot_method = solver.plot_log if log_plot else solver.plot

    figures_before = set(plt.get_fignums())
    original_show = plt.show

    try:
        plt.show = lambda *args, **kwargs: None
        with redirect_stdout(StringIO()):
            returned = plot_method()
    finally:
        plt.show = original_show

    figures = [
        plt.figure(number) for number in plt.get_fignums() if number not in figures_before
    ]
    for candidate in (returned, getattr(returned, "figure", None)):
        if isinstance(candidate, Figure) and candidate not in figures:
            figures.append(candidate)

    if not figures:
        method_name = "plot_log()" if log_plot else "plot()"
        raise RuntimeError(
            f"extraplus.{method_name} did not return or create a Matplotlib Figure."
        )
    return figures


# The fitted models as extraplus defines them: y = C + A * phi(t) with
# t = x / x_max (extraplus._compute_basis), shown in the plot titles.
_MODEL_FORMULAS = {
    _EXPONENTIAL: r"$y = A\,e^{-B\,x/x_{\max}} + C$",
    _SQRT_EXPONENTIAL: r"$y = A\,e^{-B\sqrt{x/x_{\max}}} + C$",
    _POWER_LAW: r"$y = A\,(x/x_{\max})^{-B} + C$",
}

# Title marker of the full-range figure of extraplus.plot(), and the label
# extraplus.plot_log() gives its prediction-interval band.
_FULL_VIEW_TITLE = "Full View"
_PI_BAND_LABEL = "95% PI Band"


def _full_view_figure(figures: list[Figure]) -> Figure:
    """The full-range figure among those extraplus.plot() created."""
    for fig in figures:
        if fig.axes and _FULL_VIEW_TITLE in fig.axes[0].get_title():
            return fig
    raise RuntimeError(f"extraplus.plot() did not create its '{_FULL_VIEW_TITLE}' figure.")


def _asinh_y_axis(ax, center: float, width: float, low: float, high: float) -> None:
    """
    Show the y axis on an asinh scale around ``center``: linear within about
    +-width, logarithmic in the distance from center beyond that.  This keeps
    the whole data range and a band of half-width ``width`` around center
    readable in one plot.  Ticks sit at center and center +- 10^k (and 2, 5 x 10^k if few).
    """
    def forward(values):
        return np.arcsinh((np.asarray(values, dtype=float) - center) / width)

    def inverse(values):
        return center + width * np.sinh(np.asarray(values, dtype=float))

    ax.set_yscale("function", functions=(forward, inverse))
    u_low, u_high = float(forward(low)), float(forward(high))
    pad = 0.05 * ((u_high - u_low) or 1.0)
    ax.set_ylim(float(inverse(u_low - pad)), float(inverse(u_high + pad)))

    # At most 5 decades (a near-exact fit makes width tiny); with 2 or fewer,
    # also 2 and 5 times 10^k so the axis keeps a few labels.
    k_max = int(np.floor(np.log10(max(high - center, center - low, width))))
    k_min = max(min(int(np.ceil(np.log10(width))), k_max), k_max - 4)
    steps = (1.0,) if k_max - k_min >= 2 else (1.0, 2.0, 5.0)
    ticks = [center] + [
        center + sign * step * 10.0**k
        for k in range(k_min, k_max + 1)
        for step in steps
        for sign in (1.0, -1.0)
        if low <= center + sign * step * 10.0**k <= high
    ]
    decimals = max(0, 1 - k_min)
    ax.yaxis.set_major_locator(FixedLocator(sorted(ticks)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.{decimals}f}"))
    ax.yaxis.set_minor_locator(NullLocator())


def _style_fit_plot(fig: Figure, solver, model_type: str, x_all, y_all) -> None:
    """
    Turn extraplus.plot()'s full-range figure into the EXTRAP*_PLOT picture:
    the model formula and fitted values as title, every input point (rows
    beyond n_fit as open markers), and the C +- uncertainty band kept readable
    next to the full data range with an asinh y axis around C.
    """
    ax = fig.axes[0]
    result = solver.results[model_type]
    A = solver.y_range * float(result["A"])
    B = float(result["B"])
    C = solver.y_min + solver.y_range * float(result["C"])
    # The band extraplus.plot() draws uses sigma_mc.
    sigma = solver.y_range * float(result.get("sigma_mc", 0.0))
    sigma_ok = bool(np.isfinite(sigma) and sigma > 0.0)
    n_fitted = len(solver.raw_x)

    for line in ax.get_lines():
        label = line.get_label()
        if label == "Data":
            if n_fitted < len(x_all):
                line.set_label("Data (fitted)")
        elif not label.startswith("_"):
            line.set_label("Fit")
    if n_fitted < len(x_all):
        ax.plot(x_all[n_fitted:], y_all[n_fitted:], "o", mfc="none", mec="k",
                markersize=6, zorder=5, label="Data (not fitted)")

    for band in list(ax.collections):   # the C +- sigma band
        if sigma_ok:
            band.set_label("C ± uncertainty")
        else:
            band.remove()

    curves = [np.asarray(line.get_ydata(), dtype=float)
              for line in ax.get_lines() if line.get_label() == "Fit"]
    values = np.concatenate([np.asarray(y_all, dtype=float), [C], *curves]
                            + ([[C - sigma, C + sigma]] if sigma_ok else []))
    values = values[np.isfinite(values)]
    if sigma_ok:
        width = sigma
    else:
        distances = np.abs(np.asarray(y_all, dtype=float) - C)
        distances = distances[distances > 0.0]
        width = float(distances.min()) if distances.size else 1e-3 * solver.y_range
    _asinh_y_axis(ax, C, width, float(values.min()), float(values.max()))

    ax.set_title(
        f"{_MODEL_FORMULAS[model_type]}\n"
        f"A = {A:.6g},  B = {B:.6g},  C = {C:.10g} ± {sigma:.2g},  "
        f"x_max = {solver.x_max:g}"
    )
    ax.legend()
    fig.tight_layout()


def _style_log_plot(fig: Figure, model_type: str) -> None:
    """
    Turn extraplus.plot_log()'s figure into the EXTRAP*_PLOT_LOG picture: the
    model formula as title and no prediction-interval band (the confidence
    band stays), with the y limits fitted to what remains.
    """
    ax = fig.axes[0]
    for collection in list(ax.collections):
        if collection.get_label() == _PI_BAND_LABEL:
            collection.remove()

    values = [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):   # bands
            values += [path.vertices[:, 1] for path in collection.get_paths()]
        else:                                        # scatter points
            values.append(np.asarray(collection.get_offsets(), dtype=float).reshape(-1, 2)[:, 1])
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    if values.size:
        low, high = float(values.min()), float(values.max())
        pad = 0.05 * ((high - low) or 1.0)
        ax.set_ylim(low - pad, high + pad)

    linearization = ax.get_title().split(": ", 1)[-1]
    ax.set_title(f"{_MODEL_FORMULAS[model_type]}\nlinearized: {linearization}", fontsize=11)
    fig.suptitle("")
    ax.legend(fontsize=8)
    fig.tight_layout()


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
    solver, x, y = _build_fitted_solver_for_plot(
        x_range=x_range,
        y_range=y_range,
        model_type=model_type,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )

    figures = _figures_from_extraplus(solver, log_plot=log_plot)

    try:
        if log_plot:
            fig = figures[-1]
            _style_log_plot(fig, model_type)
        else:
            fig = _full_view_figure(figures)
            _style_fit_plot(fig, solver, model_type, x, y)
        _insert_plot_in_excel(
            caller=caller,
            fig=fig,
            picture_name=picture_name,
        )
    finally:
        for figure in figures:
            plt.close(figure)

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
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).extrapolated


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
@_excel_errors
def EXTRAP2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).extrapolated


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
@_excel_errors
def EXTRAP3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).extrapolated


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
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).B


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
def DECAY_RATE2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential B for sqrt(x/x_max)."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).B


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@_excel_errors
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
    return _package_version("extraplus")


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