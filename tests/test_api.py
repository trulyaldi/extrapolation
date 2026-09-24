"""Tests for the public Python API: extraplus.extrapolate(), compare() and Result."""

import dataclasses
import doctest
import io
import json
import math
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import extraplus
from extraplus import Result, compare, extrapolate
from extraplus import _result, api

MODELS = {1: "exponential", 2: "sqrt_exponential", 3: "power_law"}
RESULT_KEYS = ["value", "uncertainty", "decay_rate", "amplitude", "r2",
               "model", "n_points", "n_fit_used", "x_max", "b_fixed"]


def test_public_names():
    assert set(extraplus.__all__) == {"extrapolate", "compare", "Result", "extraplus", "__version__"}
    assert isinstance(extraplus.__version__, str) and extraplus.__version__
    assert callable(extraplus.extraplus)   # the research engine stays importable


@pytest.mark.parametrize("convert", [
    list, tuple, np.asarray, pd.Series,
    lambda v: np.asarray(v, dtype=float).reshape(-1, 1),   # an Excel column range
], ids=["list", "tuple", "ndarray", "Series", "column"])
def test_array_like_inputs_give_the_same_result(convert, be_xy):
    x, y = be_xy
    assert extrapolate(convert(x), convert(y)) == extrapolate(x, y)


def test_result_fields(be_xy):
    x, y = be_xy
    result = extrapolate(x, y)
    assert isinstance(result, Result)
    assert result.model == "exponential"
    assert result.n_points == result.n_fit_used == 12
    assert result.x_max == 1200.0
    assert result.b_fixed is None
    assert result.value == pytest.approx(-14.66735651, abs=1e-5)
    assert 0.0 < result.uncertainty < 1e-4
    assert result.decay_rate > 0 and result.amplitude > 0 and 0 < result.r2 <= 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.value = 0.0


@pytest.mark.parametrize("number, name", MODELS.items())
def test_models_by_name_and_number(number, name, be_xy):
    by_number, by_name = extrapolate(*be_xy, model=number), extrapolate(*be_xy, model=name)
    assert by_number == by_name and by_name.model == name
    assert extrapolate(*be_xy, model=np.int64(number)) == by_name


@pytest.mark.parametrize("model", ["exp", "Exponential", 0, 4, 1.0, True, None])
def test_unknown_model(model, be_xy):
    message = "model must be 'exponential', 'sqrt_exponential', 'power_law', 1, 2 or 3."
    with pytest.raises(ValueError, match=re.escape(message)):
        extrapolate(*be_xy, model=model)


def test_b_fixed(be_xy):
    result = extrapolate(*be_xy, b_fixed=5.0)
    assert result.b_fixed == 5.0 and result.decay_rate == 5.0
    assert result.value != extrapolate(*be_xy).value
    assert "fixed at 5" in result.summary()


def test_n_fit(be_xy):
    result = extrapolate(*be_xy, n_fit=5)
    assert (result.n_points, result.n_fit_used, result.x_max) == (12, 5, 1200.0)
    assert extrapolate(*be_xy, n_fit=5.7) == result          # truncated to 5
    assert extrapolate(*be_xy, n_fit=100) == extrapolate(*be_xy)   # capped at 12


def test_unusable_pairs_are_skipped(be_xy):
    x, y = be_xy
    padded_x = ["Basis Size", *x, None, "", 1300, float("nan")]
    padded_y = ["Energy", *y, None, "", "n/a", 1.0]
    result = extrapolate(padded_x, padded_y)
    assert result == extrapolate(x, y) and result.n_points == 12


@pytest.mark.parametrize("x, y, kwargs, message", [
    ([100, 200, 300], [1.0, 2.0], {}, "x and y must have the same length."),
    ([100, 200], [1.0, 2.0], {}, "At least three paired numeric x/y observations are required."),
    ([5, 5, 5], [1.0, 2.0, 3.0], {}, "x must contain at least two distinct values."),
    ([-2, -1, 0], [1.0, 2.0, 3.0], {}, "The largest x value cannot be zero because x is scaled by x_max."),
    ([100, 200, 300], [1.0, 1.0, 1.0], {}, "y is constant, so an extrapolation cannot be fitted."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"b_fixed": -1.0}, "b_fixed must be greater than zero."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"b_fixed": float("inf")}, "b_fixed must be finite."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"b_fixed": "abc"}, "b_fixed must be a number."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"n_fit": 2}, "n_fit must be at least 3."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"n_fit": float("nan")}, "n_fit must be finite."),
    ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], {"n_fit": "abc"}, "n_fit must be a number."),
])
def test_invalid_input_raises_value_error(x, y, kwargs, message):
    with pytest.raises(ValueError) as info:
        extrapolate(x, y, **kwargs)
    assert str(info.value) == message
    with pytest.raises(ValueError, match=re.escape(message)):
        compare(x, y, **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"b_fixed": 5.0}, {"n_fit": 6}, {"b_fixed": 4.0, "n_fit": 7}],
                         ids=["default", "b_fixed", "n_fit", "both"])
@pytest.mark.parametrize("rising", [False, True], ids=["falling", "rising"])
@pytest.mark.parametrize("model", MODELS.values())
def test_predict_equals_the_engine_fitted_values(model, rising, kwargs, be_xy):
    x, y = be_xy
    if rising:
        y = [-v for v in y]
    result = extrapolate(x, y, model, **kwargs)
    solver = result._solver()
    fitted = solver.y_min + solver.y_range * solver.results[model]["y_pred"]
    x_fit = np.asarray(x[:result.n_fit_used], dtype=float)
    np.testing.assert_allclose(result.predict(x_fit), fitted, rtol=1e-12, atol=0)
    assert [result.predict(v) for v in x_fit] == result.predict(x_fit).tolist()


@pytest.mark.parametrize("kwargs", [{}, {"b_fixed": 5.0}, {"n_fit": 6}], ids=["default", "b_fixed", "n_fit"])
@pytest.mark.parametrize("model", ["exponential", "sqrt_exponential"])
def test_predict_tends_to_the_extrapolated_value(model, kwargs, be_xy):
    result = extrapolate(*be_xy, model, **kwargs)
    gaps = [abs(result.predict(x) - result.value) for x in (1300, 2000, 5000, 20000)]
    assert gaps == sorted(gaps, reverse=True) and gaps[0] > 0
    assert result.predict(1e300) == result.value


def test_predict_returns_a_float_or_an_array_of_the_input_shape(be_xy):
    result = extrapolate(*be_xy)
    value = result.predict(1500)
    assert type(value) is float
    assert result.predict(np.float64(1500)) == result.predict("1500") == value
    assert type(result.predict(np.asarray(1500.0))) is float
    targets = [1300, 1500, 2000]
    expected = [result.predict(v) for v in targets]
    for convert in (list, tuple, np.asarray, pd.Series):
        array = result.predict(convert(targets))
        assert isinstance(array, np.ndarray) and array.shape == (3,) and array.tolist() == expected
    assert result.predict(np.reshape(targets, (3, 1))).tolist() == [[v] for v in expected]
    assert result.predict([]).shape == (0,)


@pytest.mark.parametrize("target, message", [
    (None, "target is required."),
    ("", "target is required."),
    (0, "target must be greater than zero."),
    (-1500.0, "target must be greater than zero."),
    ("abc", "target must be a number."),
    (float("inf"), "target must be finite."),
    (float("-inf"), "target must be finite."),
    (float("nan"), "target must be finite."),
    ([1500, 0], "target must be greater than zero."),
])
def test_predict_rejects_invalid_targets(target, message, be_xy):
    result = extrapolate(*be_xy)
    with pytest.raises(ValueError) as info:
        result.predict(target)
    assert str(info.value) == message


def test_summary(be_xy):
    result = extrapolate(*be_xy, model="power_law")
    text = result.summary()
    assert isinstance(text, str)
    assert text.splitlines()[0] == "Extrapolation with the power_law model, y = A * (x/x_max)^(-B) + C"
    assert f"{result.value:.15g}" in text and "(fitted)" in text and "12 (12 fitted)" in text


def test_to_dict(be_xy):
    result = extrapolate(*be_xy, b_fixed=4.0, n_fit=6)
    data = result.to_dict()
    assert list(data) == RESULT_KEYS
    assert data == {key: getattr(result, key) for key in RESULT_KEYS}
    assert data["b_fixed"] == 4.0 and data["n_fit_used"] == 6
    assert json.loads(json.dumps(data)) == data   # plain Python values


def test_compare(be_xy):
    table = compare(*be_xy, n_fit=8)
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == list(MODELS.values())
    assert list(table.columns) == [key for key in RESULT_KEYS if key != "model"]
    for name in MODELS.values():
        expected = extrapolate(*be_xy, model=name, n_fit=8).to_dict()
        row = table.loc[name]
        for key in ("value", "uncertainty", "decay_rate", "amplitude", "r2", "x_max"):
            assert row[key] == expected[key]
        assert row["n_fit_used"] == 8


@pytest.mark.parametrize("n_fit", [None, 5, 3])
def test_warning_only_when_the_uncertainty_is_not_finite(n_fit, be_xy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extrapolate(*be_xy, n_fit=n_fit)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert bool(runtime) == (not math.isfinite(result.uncertainty))
    if runtime:
        assert "not finite" in str(runtime[0].message)
        assert runtime[0].filename == __file__   # points at the caller


@pytest.mark.parametrize("method", ["plot", "plot_linearized"])
def test_plots_return_one_open_figure(method, be_xy):
    before = set(plt.get_fignums())
    fig = getattr(extrapolate(*be_xy, n_fit=6), method)()
    try:
        assert isinstance(fig, Figure)
        assert set(plt.get_fignums()) - before == {fig.number}   # nothing else left open
        fig.savefig(io.BytesIO(), format="png")
    finally:
        plt.close(fig)


def test_nothing_is_printed(capsys, be_xy):
    result = extrapolate(*be_xy, model=2, n_fit=6)
    compare(*be_xy)
    result.summary()
    result.to_dict()
    for fig in (result.plot(), result.plot_linearized()):
        plt.close(fig)
    with pytest.raises(ValueError):
        extrapolate([1, 2], [3, 4])
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("module", [api, _result])
def test_docstring_examples(module):
    failures, tried = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert tried > 0 and failures == 0
