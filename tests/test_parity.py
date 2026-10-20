"""Each Excel UDF equals the matching Python API call (values, errors and plots)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "excel"))

import extrapolation  # noqa: E402
from extraplus import extrapolate  # noqa: E402

MODELS = {1: "exponential", 2: "sqrt_exponential", 3: "power_law"}


def column(values):
    """Shape a list like an Excel column range passed with np.array, ndim=2."""
    return np.asarray(values, dtype=object).reshape(-1, 1)


@pytest.fixture(params=["falling", "rising", "padded"])
def data(request, be_xy):
    x, y = be_xy
    if request.param == "rising":
        y = [-v for v in y]
    elif request.param == "padded":
        x, y = ["Basis Size", *x, None, ""], ["Energy", *y, None, 1.0]
    return x, y


@pytest.mark.parametrize("args", [(), (5.0,), (None, 6), (4.0, 7.9)], ids=["default", "b_fixed", "n_fit", "both"])
@pytest.mark.parametrize("n", MODELS)
def test_numeric_udfs_equal_the_api(n, args, data):
    x, y = data
    result = extrapolate(x, y, MODELS[n], *args)
    X, Y = column(x), column(y)
    assert getattr(extrapolation, f"EXTRAP{n}")(X, Y, *args) == result.value
    assert getattr(extrapolation, f"UNCERTAINTY{n}")(X, Y, *args) == result.uncertainty
    assert getattr(extrapolation, f"DECAY_RATE{n}")(X, Y, *args) == result.decay_rate
    triple = [result.value, result.uncertainty, result.decay_rate]
    assert getattr(extrapolation, f"EXTRAP{n}_V")(X, Y, *args).tolist() == [[v] for v in triple]
    assert getattr(extrapolation, f"EXTRAP{n}_H")(X, Y, *args).tolist() == [triple]


@pytest.mark.parametrize("args", [(), (5.0,), (None, 6), (4.0, 7.9)], ids=["default", "b_fixed", "n_fit", "both"])
@pytest.mark.parametrize("n", MODELS)
def test_finite_udfs_equal_predict(n, args, data):
    x, y = data
    result = extrapolate(x, y, MODELS[n], *args)
    udf = getattr(extrapolation, f"FINITE_EXTRAP{n}")
    for target in (15000, 1300.0, 650, 1e300):
        assert udf(column(x), column(y), target, *args) == result.predict(target)


@pytest.mark.parametrize("target", [None, "", 0, -1500.0, "abc", float("inf"), float("nan")])
@pytest.mark.parametrize("n", MODELS)
def test_finite_udf_target_errors_equal_the_api_message(n, target, be_xy):
    x, y = be_xy
    with pytest.raises(ValueError) as info:
        extrapolate(x, y, MODELS[n]).predict(target)
    udf = getattr(extrapolation, f"FINITE_EXTRAP{n}")
    assert udf(column(x), column(y), target) == f"Error: {info.value}"
    if target is None:   # left out of the formula
        assert udf(column(x), column(y)) == "Error: target is required."


@pytest.mark.parametrize("x, y, args", [
    ([100, 200, 300], [1.0, 2.0], ()),
    ([100, 200], [1.0, 2.0], ()),
    ([100, 200, 300], [1.0, 1.0, 1.0], ()),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], ("abc",)),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], (None, float("inf"))),
])
@pytest.mark.parametrize("n", MODELS)
def test_error_texts_equal_the_api_message(n, x, y, args):
    with pytest.raises(ValueError) as info:
        extrapolate(x, y, MODELS[n], *args)
    for name in (f"EXTRAP{n}", f"UNCERTAINTY{n}", f"DECAY_RATE{n}"):
        assert getattr(extrapolation, name)(column(x), column(y), *args) == f"Error: {info.value}"
    # The spill functions put the same text in each of their 3 cells.
    assert getattr(extrapolation, f"EXTRAP{n}_V")(column(x), column(y), *args).tolist() == \
        [[f"Error: {info.value}"]] * 3
    assert getattr(extrapolation, f"EXTRAP{n}_H")(column(x), column(y), *args).tolist() == \
        [[f"Error: {info.value}"] * 3]
    assert getattr(extrapolation, f"FINITE_EXTRAP{n}")(column(x), column(y), 1500, *args) == f"Error: {info.value}"


class _Pictures:
    def add(self, fig, **kwargs):
        self.fig = fig


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.mark.parametrize("suffix, method", [("", "plot"), ("_LOG", "plot_linearized")])
@pytest.mark.parametrize("n", MODELS)
def test_plot_udfs_draw_the_api_figure(n, suffix, method, be_xy):
    x, y = be_xy
    caller = SimpleNamespace(row=1, column=1, left=0.0, top=0.0, height=15.0,
                             sheet=SimpleNamespace(pictures=_Pictures()))
    getattr(extrapolation, f"EXTRAP{n}_PLOT{suffix}")(column(x), column(y), caller, None, 6)
    fig = getattr(extrapolate(x, y, MODELS[n], n_fit=6), method)()
    try:
        assert np.array_equal(_pixels(caller.sheet.pictures.fig), _pixels(fig))
    finally:
        plt.close(fig)
