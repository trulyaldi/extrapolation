"""
Tests for the Excel UDF wrappers in excel/extrapolation.py.

They call the Python functions behind the Excel formulas directly (no Excel
needed) and check that they still work with the current extraplus core.  Their
main job is to catch a mismatch between the wrappers and the core, for example
a change to the arguments of ``extraplus.compute_uncertainty()``.
"""

import importlib.metadata
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")   # before extrapolation imports pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "excel"))

import extrapolation  # noqa: E402
from extraplus import _plotting, api, extraplus  # noqa: E402

# Be 1Se ground-state energy, first 12 basis sizes of
# research/data/large-dataset/Be_1Se.csv (the converged value is -14.6673565...).
BE_1SE_ENERGY = [
    (100, -14.6661876127332),
    (200, -14.6667402339119),
    (300, -14.6672828224583),
    (400, -14.6673249180953),
    (500, -14.66734057696),
    (600, -14.6673445259951),
    (700, -14.6673478029866),
    (800, -14.6673503395393),
    (900, -14.6673519816691),
    (1000, -14.6673531093866),
    (1100, -14.6673536576132),
    (1200, -14.6673541629258),
]

MODELS = {1: "exponential", 2: "sqrt_exponential", 3: "power_law"}
SCALAR_UDFS = [f"{base}{n}" for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE") for n in MODELS]
SPILL_UDFS = [f"EXTRAP{n}{suffix}" for suffix in ("_V", "_H") for n in MODELS]
PLOT_UDFS = [f"EXTRAP{n}_PLOT{suffix}" for suffix in ("", "_LOG") for n in MODELS]


def excel_column(values):
    """Shape a list like an Excel column range passed with np.array, ndim=2."""
    return np.asarray(values, dtype=object).reshape(-1, 1)


@pytest.fixture
def xy():
    x, y = zip(*BE_1SE_ENERGY)
    return excel_column(x), excel_column(y)


@pytest.mark.parametrize("name", SCALAR_UDFS)
def test_scalar_udfs_return_finite_numbers(name, xy):
    result = getattr(extrapolation, name)(*xy)
    assert isinstance(result, float), f"{name} returned {result!r}"
    assert math.isfinite(result)


@pytest.mark.parametrize("name", SPILL_UDFS)
def test_spill_udfs_return_three_finite_numbers(name, xy):
    result = getattr(extrapolation, name)(*xy)
    assert isinstance(result, np.ndarray), f"{name} returned {result!r}"
    assert result.shape == ((3, 1) if name.endswith("_V") else (1, 3))
    assert np.all(np.isfinite(result))


def test_extrapolated_value_is_near_the_converged_energy(xy):
    # The largest basis (16000) gives -14.6673565083605.
    for n in MODELS:
        assert getattr(extrapolation, f"EXTRAP{n}")(*xy) == pytest.approx(-14.66735651, abs=1e-5)


@pytest.mark.parametrize("n", MODELS)
def test_wrapper_matches_extraplus_called_directly(n, xy):
    """The wrapper must only rescale what extraplus computes, with the Excel defaults."""
    x, y = (np.asarray(col, dtype=float).ravel() for col in xy)
    solver = extraplus(df=pd.DataFrame({"x": x, "y": y}), x_col="x", y_col="y",
                       err_df=None, inf_df=None, b_init=None, n_fit=None, use_energy_b=False)
    solver.fit_linearized(models=[MODELS[n]], verbose=False, compute_uq=False)
    solver.compute_uncertainty()
    res = solver.results[MODELS[n]]

    assert getattr(extrapolation, f"EXTRAP{n}")(*xy) == solver.y_min + solver.y_range * res["C"]
    assert getattr(extrapolation, f"UNCERTAINTY{n}")(*xy) == solver.y_range * res["sigma_C"]
    assert getattr(extrapolation, f"DECAY_RATE{n}")(*xy) == res["B"]


def test_version_udf():
    assert extrapolation.EXTRAPLUS_VERSION() == importlib.metadata.version("extraplus")


@pytest.mark.parametrize(
    "x, y, b_fixed, expected",
    [
        ([100, 200, 300], [1.0, 2.0], None, "x and y must have the same length"),
        ([100, 200], [1.0, 2.0], None, "At least three paired numeric"),
        ([100, 200, 300], [1.0, 1.0, 1.0], None, "y is constant"),
        ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], -1.0, "b_fixed must be greater than zero"),
        ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], "abc", "b_fixed must be a number"),
    ],
)
def test_bad_input_returns_error_text(x, y, b_fixed, expected):
    for name in SCALAR_UDFS + SPILL_UDFS:
        result = getattr(extrapolation, name)(excel_column(x), excel_column(y), b_fixed)
        assert isinstance(result, str) and result.startswith("Error: "), f"{name}: {result!r}"
        assert expected in result, f"{name}: {result!r}"


def test_blank_and_text_cells_are_ignored(xy):
    x, y = xy
    padded_x = excel_column(["Basis Size", *x.ravel(), None, ""])
    padded_y = excel_column(["Energy", *y.ravel(), None, ""])
    assert extrapolation.EXTRAP1(padded_x, padded_y) == extrapolation.EXTRAP1(x, y)


# ---------------------------------------------------------------------------
# Plot UDFs
# ---------------------------------------------------------------------------


class _Pictures:
    """Stands in for xlwings' sheet.pictures; keeps the figure it is given."""

    def add(self, fig, **kwargs):
        self.fig = fig


def run_plot_udf(name, x, y, *args):
    """Call a _PLOT UDF with a fake calling cell; return its status and figure."""
    caller = SimpleNamespace(row=2, column=4, left=0.0, top=0.0, height=15.0,
                             sheet=SimpleNamespace(pictures=_Pictures()))
    figures_before = set(plt.get_fignums())
    status = getattr(extrapolation, name)(x, y, caller, *args)
    assert set(plt.get_fignums()) == figures_before, "the UDF left pyplot figures open"
    return status, caller.sheet.pictures.fig


@pytest.mark.parametrize("n_fit", [None, 5, 3])   # n_fit=3 has no residual dof: sigma = inf on dev
@pytest.mark.parametrize("name", PLOT_UDFS)
def test_plot_udfs_draw_a_figure(name, n_fit, xy):
    status, fig = run_plot_udf(name, *xy, None, n_fit)
    assert status == f"{name} updated"
    assert isinstance(fig, Figure)
    fig.savefig(io.BytesIO(), format="png", bbox_inches="tight", dpi=50)   # as xlwings saves it


@pytest.mark.parametrize("name", PLOT_UDFS)
def test_plot_udfs_draw_rising_data(name, xy):
    x, y = xy
    rising = excel_column([-value for value in y.ravel()])
    status, fig = run_plot_udf(name, x, rising)
    assert status == f"{name} updated"
    fig.savefig(io.BytesIO(), format="png", bbox_inches="tight", dpi=50)


def test_plot_titles_show_each_model_formula(xy):
    for suffix in ("", "_LOG"):
        titles = [run_plot_udf(f"EXTRAP{n}_PLOT{suffix}", *xy)[1].axes[0].get_title().split("\n")[0]
                  for n in MODELS]
        assert titles == [_plotting.MODEL_FORMULAS[MODELS[n]] for n in MODELS]
        assert len(set(titles)) == 3


@pytest.mark.parametrize("n", MODELS)
def test_fit_plot_shows_all_data_and_the_extrapolation_range(n, xy):
    x = np.asarray(xy[0], dtype=float).ravel()
    for n_fit in (None, 5):
        ax = run_plot_udf(f"EXTRAP{n}_PLOT", *xy, None, n_fit)[1].axes[0]
        plotted_x = np.concatenate([line.get_xdata() for line in ax.get_lines()
                                    if line.get_label().startswith("Data")])
        assert sorted(plotted_x) == sorted(x)
        low, high = ax.get_xlim()
        assert low <= x.min() and high >= 1.5 * x.max()
        assert "C ± uncertainty" in [t.get_text() for t in ax.get_legend().get_texts()]


@pytest.mark.parametrize("n", MODELS)
def test_log_plot_has_the_confidence_band_but_no_prediction_band(n, xy):
    ax = run_plot_udf(f"EXTRAP{n}_PLOT_LOG", *xy)[1].axes[0]
    labels = [c.get_label() for c in ax.collections]
    assert "95% CI Band" in labels
    assert not any("PI" in label for label in labels)
    assert not any("PI" in t.get_text() for t in ax.get_legend().get_texts())

    # The y range is fitted to what remains, not to extraplus' own PI band.
    x, y = (np.asarray(col, dtype=float).ravel() for col in xy)
    solver = api._solve(x, y, MODELS[n], None, None)
    original = _plotting.figures_from_extraplus(solver, log_plot=True)
    try:
        (pi,) = [c for c in original[-1].axes[0].collections if c.get_label() == "95% PI Band"]
        pi_y = np.concatenate([path.vertices[:, 1] for path in pi.get_paths()])
    finally:
        for fig in original:
            plt.close(fig)
    low, high = ax.get_ylim()
    assert low > pi_y.min() or high < pi_y.max()
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    band_y = np.concatenate([path.vertices[:, 1] for c in bands for path in c.get_paths()])
    assert low <= band_y.min() and high >= band_y.max()
