"""EXTRAP_ALL_V: column n must be exactly EXTRAPn_V, bit for bit, for the same input."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "excel"))

import extrapolation  # noqa: E402
from udf_cases import DATASETS, FAILING, bits, column  # noqa: E402

MODELS = (1, 2, 3)


def singles(X, Y, b_values=(None, None, None)):
    return [getattr(extrapolation, f"EXTRAP{n}_V")(X, Y, b) for n, b in zip(MODELS, b_values)]


@pytest.mark.parametrize("name", [n for n, case in DATASETS.items() if case[2] in (None, "") and case[3] in (None, "")])
def test_columns_equal_the_three_single_functions(name):
    x, y, *_ = DATASETS[name]
    X, Y = column(x), column(y)
    result = extrapolation.EXTRAP_ALL_V(X, Y)
    assert isinstance(result, np.ndarray) and result.shape == (3, 3) and result.dtype == float
    for n, single in zip(MODELS, singles(X, Y)):
        assert bits(result[:, n - 1:n].copy()) == bits(single), (name, n)


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_b_fixed_per_model_from_another_call(shape, be_xy):
    """The chain: the B row of the energy's EXTRAP_ALL_V fixes B for a property."""
    X, Y = column(be_xy[0]), column(be_xy[1])
    energy = extrapolation.EXTRAP_ALL_V(X, Y)
    b_row = energy[2:3, :].reshape(shape).tolist()   # as xlwings reads 3 cells with ndim=2
    prop_y = column([v * 1.5 + 0.1 for v in be_xy[1]])
    result = extrapolation.EXTRAP_ALL_V(X, prop_y, b_row)
    for n, single in zip(MODELS, singles(X, prop_y, energy[2])):
        assert bits(result[:, n - 1:n].copy()) == bits(single)
    assert result[2].tolist() == energy[2].tolist()   # B is the fixed one


def test_a_blank_cell_fits_that_models_b(be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    b = [[5.0, None, 2.0]]   # a blank cell is None in a raw (not np.array) read
    result = extrapolation.EXTRAP_ALL_V(X, Y, b)
    for n, single in zip(MODELS, singles(X, Y, (5.0, None, 2.0))):
        assert bits(result[:, n - 1:n].copy()) == bits(single)


@pytest.mark.parametrize("blank", [None, [[None]], [[""]]])
def test_blank_b_fixed_fits_all(blank, be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    assert bits(extrapolation.EXTRAP_ALL_V(X, Y, blank)) == bits(extrapolation.EXTRAP_ALL_V(X, Y))


@pytest.mark.parametrize("b", [[[2.0]], [[1.0, 2.0]], [[1.0, 2.0, 3.0, 4.0]]])
def test_b_fixed_of_another_size_is_an_error_in_every_cell(b, be_xy):
    result = extrapolation.EXTRAP_ALL_V(column(be_xy[0]), column(be_xy[1]), b)
    assert result.shape == (3, 3)
    assert set(result.ravel()) == {"Error: b_fixed must be blank or 3 cells, one B for each model (1, 2, 3)."}


@pytest.mark.parametrize("name", FAILING)
def test_input_errors_fill_every_cell_with_the_single_functions_text(name):
    x, y, *_ = FAILING[name]
    X, Y = column(x), column(y)
    result = extrapolation.EXTRAP_ALL_V(X, Y)
    if name in ("bad_b_fixed", "text_n_fit"):   # those inputs are in b_fixed/n_fit, not given here
        assert result.dtype == float
        return
    assert result.shape == (3, 3)
    assert result.ravel().tolist() == [extrapolation.EXTRAP1(X, Y)] * 9


def test_one_failing_model_does_not_hide_the_others(be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    b = [[5.0, "#N/A", None]]   # an Excel error in model 2's B cell (err_to_str)
    result = extrapolation.EXTRAP_ALL_V(X, Y, b)
    assert result[:, 1].tolist() == ["Error: b_fixed must be a number."] * 3
    for n, b_n in ((1, 5.0), (3, None)):
        single = getattr(extrapolation, f"EXTRAP{n}_V")(X, Y, b_n)
        assert [float(v) for v in result[:, n - 1]] == single.ravel().tolist()
