"""
Plots of one fitted model, drawn by extraplus.plot() / plot_log() and then
restyled: the model formula as title, the whole data range with a readable
uncertainty band (fit plot), and no prediction-interval band (linearized plot).

Used by ``Result.plot()`` / ``Result.plot_linearized()`` and by the Excel
EXTRAP*_PLOT / EXTRAP*_PLOT_LOG functions.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# The fitted models as extraplus defines them: y = C + A * phi(t) with
# t = x / x_max (extraplus._compute_basis), shown in the plot titles.
MODEL_FORMULAS = {
    "exponential": r"$y = A\,e^{-B\,x/x_{\max}} + C$",
    "sqrt_exponential": r"$y = A\,e^{-B\sqrt{x/x_{\max}}} + C$",
    "power_law": r"$y = A\,(x/x_{\max})^{-B} + C$",
}

# Title marker of the full-range figure of extraplus.plot(), and the label
# extraplus.plot_log() gives its prediction-interval band.
_FULL_VIEW_TITLE = "Full View"
_PI_BAND_LABEL = "95% PI Band"


def fit_figure(solver, model: str, x_all: np.ndarray, y_all: np.ndarray) -> Figure:
    """
    The fit plot of ``model``: every point of ``x_all``/``y_all`` (rows beyond
    n_fit as open markers), the fitted curve out to 1.5 x_max, and C +- its
    uncertainty.  The returned figure is open in pyplot; the others that
    extraplus.plot() creates are closed.
    """
    figures = figures_from_extraplus(solver, log_plot=False)
    try:
        fig = _full_view_figure(figures)
        _style_fit_plot(fig, solver, model, x_all, y_all)
    except BaseException:
        for figure in figures:
            plt.close(figure)
        raise
    for figure in figures:
        if figure is not fig:
            plt.close(figure)
    return fig


def linearized_figure(solver, model: str) -> Figure:
    """
    The linearized-space plot of ``model`` with its 95% confidence band (the
    prediction band of extraplus.plot_log() is removed).  The returned figure
    is open in pyplot; any others are closed.
    """
    figures = figures_from_extraplus(solver, log_plot=True)
    fig = figures[-1]
    try:
        _style_log_plot(fig, model)
    except BaseException:
        for figure in figures:
            plt.close(figure)
        raise
    for figure in figures:
        if figure is not fig:
            plt.close(figure)
    return fig


def figures_from_extraplus(solver, log_plot: bool) -> list[Figure]:
    """
    Call extrap.py's existing plot() or plot_log() method and return every
    Matplotlib Figure it created, oldest first.

    This supports plotting methods that return a Figure (or an Axes) as well as
    ones that only create pyplot figures and return None.  plt.show() is
    suppressed and the printed summary table is discarded, because callers need
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
    Turn extraplus.plot()'s full-range figure into the fit plot: the model
    formula and fitted values as title, every input point (rows beyond n_fit
    as open markers), and the C +- uncertainty band kept readable next to the
    full data range with an asinh y axis around C.
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
        f"{MODEL_FORMULAS[model_type]}\n"
        f"A = {A:.6g},  B = {B:.6g},  C = {C:.10g} ± {sigma:.2g},  "
        f"x_max = {solver.x_max:g}"
    )
    ax.legend()
    fig.tight_layout()


def _style_log_plot(fig: Figure, model_type: str) -> None:
    """
    Turn extraplus.plot_log()'s figure into the linearized plot: the model
    formula as title and no prediction-interval band (the confidence band
    stays), with the y limits fitted to what remains.
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
    ax.set_title(f"{MODEL_FORMULAS[model_type]}\nlinearized: {linearization}", fontsize=11)
    fig.suptitle("")
    ax.legend(fontsize=8)
    fig.tight_layout()
