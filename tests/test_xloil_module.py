"""
excel/extraplus_xloil.py: the worksheet functions copied from extrapolation.py, with xlOil's
func / arg / ret decorators.  Without xlOil (here) the decorators only record the registration.

- The recorded registration (argument names, read options, help, optional, ret options) is the
  xlwings one (extraplus_xlcompat.SIGNATURES, itself checked against xlwings in
  test_xloil_compat.py, and the ret options from test_registration's dump).
- Every numeric function returns the same bits as the xlwings module's function for every test
  dataset (same Python values in, so this checks the copied bodies).
- The size used for xlwings' expand="table" resize.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "excel"))

import extraplus_xlcompat as c  # noqa: E402
import extraplus_xloil as xo  # noqa: E402
import extrapolation  # noqa: E402
from test_registration import dump  # noqa: E402,F401  (fixture)
from udf_cases import DATASETS, FAILING, FINITE_TARGETS, column  # noqa: E402


def bits(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("object", value.shape, tuple(bits(v) for v in value.ravel()))
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    return (type(value).__name__, value)


def test_not_registered_with_excel_here():
    assert xo.xlo is None


def test_every_function_is_defined_and_recorded():
    assert set(xo.REGISTERED) == set(extrapolation.__all__) == set(c.SIGNATURES)
    for name in extrapolation.__all__:
        assert callable(getattr(xo, name)), name


def test_registration_is_the_xlwings_one():
    for name, spec in c.SIGNATURES.items():
        assert xo.REGISTERED[name]["args"] == spec, name
        assert xo.REGISTERED[name]["category"] == "Extrapolation", name
        assert xo.REGISTERED[name]["volatile"] is False, name


def test_expand_is_where_xlwings_has_it(dump):  # noqa: F811
    for name, info in dump.items():
        if name.startswith("_"):
            continue
        expected = {k: v for k, v in info["ret"].items() if k == "expand"}
        assert {k: v for k, v in xo.REGISTERED[name]["ret"].items() if k == "expand"} == expected, name
    assert sorted(n for n, r in xo.REGISTERED.items() if r["ret"].get("expand")) == sorted(
        f"EXTRAP{n}{s}" for s in ("_V", "_H") for n in (1, 2, 3))


NUMERIC = [n for n in extrapolation.__all__ if not n.endswith(("_PLOT", "_PLOT_LOG"))]


@pytest.mark.parametrize("dataset", list(DATASETS) + list(FAILING))
@pytest.mark.parametrize("name", NUMERIC)
def test_same_results_as_the_xlwings_module(name, dataset, monkeypatch):
    monkeypatch.setattr(extrapolation, "_CACHE", extrapolation._ResultCache({"EXTRAPLUS_CACHE": "off"}))
    monkeypatch.setattr(xo, "_CACHE", extrapolation._CACHE)
    x, y, b_fixed, n_fit = DATASETS.get(dataset) or FAILING[dataset]
    X, Y = column(x), column(y)
    if name in ("EXTRAPLUS_VERSION", "EXTRAPLUS_CACHE_STATS"):
        calls = [()]
    elif name.startswith("FINITE_"):
        calls = [(X, Y, t, b_fixed, n_fit) for t in FINITE_TARGETS]
    elif name == "EXTRAP_ALL_V":
        calls = [(X, Y), (X, Y, b_fixed)]
    else:
        calls = [(X, Y, b_fixed, n_fit), (X, Y)]
    for args in calls:
        assert bits(getattr(xo, name)(*args)) == bits(getattr(extrapolation, name)(*args)), (name, args[2:])


@pytest.mark.parametrize("result, size", [
    (np.zeros((3, 1)), (3, 1)),
    (np.zeros((1, 3)), (1, 3)),
    (np.zeros((3, 3)), (3, 3)),
    (np.full((3, 1), "Error: x", dtype=object), (3, 1)),
    (1.5, (1, 1)),
    ("Error: x", (1, 1)),
    ([[1.0], [2.0]], (2, 1)),
])
def test_result_size_for_the_resize(result, size):
    assert xo._result_size(result) == size
