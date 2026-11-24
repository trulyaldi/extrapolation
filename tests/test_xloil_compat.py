"""
excel/extraplus_xlcompat.py against xlwings itself, so the xlOil add-in gives
every function exactly the values xlwings gives it and shows exactly what
xlwings shows.

- Signatures: argument names, optional flags, help texts and read options are
  those registered with xlwings (runs everywhere, via test_registration's dump).
- Conversion (needs xlwings' Windows engine; skipped elsewhere): read() equals
  xlwings.conversion.read and written() equals xlwings' write + CallUDF, for a
  table of raw COM values.
- Outputs (Windows): for every numeric function and every test dataset, the
  result reaching Excel through extraplus_xlcompat has the same bits as through
  xlwings.
- xlOil values: xloil_raw() turns xlOil's object arrays into the COM values.
- Plots: process_image() writes the same PNG bytes as xlwings.utils.process_image.
"""

import datetime as dt
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "excel"))

import extraplus_xlcompat as c  # noqa: E402
import extrapolation  # noqa: E402
from test_registration import dump  # noqa: E402,F401  (fixture)
from udf_cases import DATASETS, FAILING  # noqa: E402


def _xlwings_conversion():
    try:
        from xlwings import conversion
        conversion.read(None, 1.0, {})
        return conversion
    except Exception:
        return None


XW = _xlwings_conversion()
needs_xlwings = pytest.mark.skipif(XW is None, reason="needs xlwings with its Windows (COM) engine")


def bits(value):
    """Exact comparable form (float bits, NaN payloads, -0.0, types, nesting)."""
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, tuple(bits(v) for v in value.ravel().tolist())
                if value.dtype == object else value.tobytes())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(bits(v) for v in value))
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    return (type(value).__name__, value)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered(request):
    """The xlwings registration: the real decorators' metadata on Windows, else test_registration's dump
    (which stubs the Windows modules, and so only works where they are missing)."""
    if XW is not None and hasattr(extrapolation.EXTRAP1, "__xlfunc__"):
        out = {}
        for name in extrapolation.__all__:
            f = getattr(extrapolation, name).__xlfunc__
            out[name] = {"args": [{"name": a["name"], "optional": "optional" in a, "doc": a["doc"],
                                   "options": {k: repr(v) for k, v in a["options"].items()}} for a in f["args"]]}
        return out
    data = request.getfixturevalue("dump")
    return {name: info for name, info in data.items() if not name.startswith("_")}


def test_signatures_are_the_xlwings_registration(registered):
    assert set(c.SIGNATURES) == set(registered)
    for name, spec in c.SIGNATURES.items():
        args = [a for a in registered[name]["args"] if a["name"] != "caller"]
        assert [a["name"] for a in args] == [s[0] for s in spec], name
        assert [not a["optional"] for a in args] == [s[3] for s in spec], name
        assert [a["doc"] or "" for a in args] == [s[2] for s in spec], name
        assert [a["options"] for a in args] == [{k: repr(v) for k, v in s[1].items()} for s in spec], name


# ---------------------------------------------------------------------------
# Conversion against xlwings
# ---------------------------------------------------------------------------

E = {code: code for code in c.CELL_ERRORS}
SPILL = -2146826238          # #SPILL!: not in xlwings' table, passes through as an int
NAN = float("nan")


def _com_dates():
    try:
        import pywintypes
        return [pywintypes.Time(dt.datetime(2024, 5, 6, 7, 8, 9))]
    except ImportError:
        return []


RAW = [
    None, "", 0.0, -0.0, 1.0, 2.5, NAN, float("inf"), True, False, "text", " ", "1.5",
    -2146826252, -2146826246, SPILL,
    ((1.0,), (2.0,), (3.0,)),
    ((1.0, 2.0, 3.0),),
    ((1.0, 2.0), (3.0, 4.0)),
    ((None,), ("",), (3.0,)),
    (("Basis Size",), (100.0,), (200.0,), (None,)),
    ((1.0,), (-2146826252,), (3.0,)),
    ((1.0,), (SPILL,), (3.0,)),
    ((True,), (2.0,)),
    ((None, None, None),),
    ((None,), (None,), (None,)),
    ((5.0, None, ""),),
] + _com_dates()

OPTIONS = [c.RANGE, c.ERRORS_AS_TEXT, c.PLAIN, {"ndim": 2, **c.ERRORS_AS_TEXT}]


@needs_xlwings
@pytest.mark.parametrize("options", OPTIONS, ids=["range", "err_to_str", "plain", "ndim2_err"])
@pytest.mark.parametrize("raw", RAW, ids=[repr(r)[:40] for r in RAW])
def test_read_equals_xlwings(raw, options):
    try:
        expected = ("ok", XW.read(None, raw, dict(options)))
    except Exception as exc:
        expected = ("raises", type(exc).__name__)
    try:
        got = ("ok", c.read(raw, dict(options)))
    except Exception as exc:
        got = ("raises", type(exc).__name__)
    assert bits(got) == bits(expected)


RESULTS = [
    1.0, -0.0, NAN, float("inf"), "Error: x", None, np.float64(2.0), 3,
    np.array([[1.0], [NAN], [float("inf")]]),
    np.array([[1.0, -0.0, 3.0]]),
    np.full((3, 1), "Error: bad", dtype=object),
    np.hstack([np.array([[1.0], [2.0], [NAN]], dtype=object), np.full((3, 1), "Error: e", dtype=object),
               np.array([[4.0], [5.0], [6.0]], dtype=object)]),
]


def _xlwings_written(ret, options):
    # xlwings.udfs.call_udf: conversion.write(ret, None, ret options); then com_server.CallUDF.
    res = XW.write(ret, None, dict(options))
    if len(res) == 1 and len(res[0]) == 1:
        res = res[0][0]
    elif len(res) == 1 and len(res[0]) > 1:
        res = res[0]
    return res


@needs_xlwings
@pytest.mark.parametrize("ret", RESULTS, ids=[repr(r)[:30] for r in RESULTS])
@pytest.mark.parametrize("options", [{}, {"expand": "table"}], ids=["plain", "expand"])
def test_written_equals_xlwings(ret, options):
    assert bits(c.written(ret)) == bits(_xlwings_written(ret, options))


# ---------------------------------------------------------------------------
# Every numeric function, every dataset
# ---------------------------------------------------------------------------


def _column_raw(values):
    """A column of cells as COM's Range.Value gives it (None for blank cells)."""
    cells = [None if (isinstance(v, float) and np.isnan(v)) else v for v in values]
    cells = [float(v) if isinstance(v, int) and not isinstance(v, bool) else v for v in cells]
    return tuple((v,) for v in cells) if len(cells) > 1 else cells[0]


def _scalar_raw(value):
    if value is None:
        return None
    return float(value) if isinstance(value, int) and not isinstance(value, bool) else value


NUMERIC = [n for n in c.SIGNATURES if not n.endswith(("_PLOT", "_PLOT_LOG")) and n not in
           ("EXTRAPLUS_VERSION", "EXTRAPLUS_CACHE_STATS")]


def _raw_args(name, data, target=15000.0):
    x, y, b_fixed, n_fit = data
    raw = {"x_range": _column_raw(x), "y_range": _column_raw(y), "b_fixed": _scalar_raw(b_fixed),
           "n_fit": _scalar_raw(n_fit), "target": target}
    if name == "EXTRAP_ALL_V" and b_fixed is not None:
        raw["b_fixed"] = ((_scalar_raw(b_fixed),), (None,), (_scalar_raw(b_fixed),))
    return [raw[arg] for arg, *_ in c.SIGNATURES[name]]


@needs_xlwings
@pytest.mark.parametrize("dataset", list(DATASETS) + list(FAILING))
@pytest.mark.parametrize("name", NUMERIC)
def test_function_results_reach_excel_with_the_same_bits(name, dataset, monkeypatch):
    monkeypatch.setattr(extrapolation, "_CACHE", extrapolation._ResultCache({"EXTRAPLUS_CACHE": "off"}))
    data = DATASETS.get(dataset) or FAILING[dataset]
    raw = _raw_args(name, data)
    options = [opt for _a, opt, *_ in c.SIGNATURES[name]]
    function = getattr(extrapolation, name)
    via_xlwings = _xlwings_written(function(*[XW.read(None, r, dict(o)) if r is not None else None
                                              for r, o in zip(raw, options)]),
                                   {"expand": "table"} if name.endswith(("_V", "_H")) else {})
    via_compat = c.written(function(*[c.read(r, o) if r is not None else None for r, o in zip(raw, options)]))
    assert bits(via_compat) == bits(via_xlwings)


# ---------------------------------------------------------------------------
# xlOil's values -> COM's values
# ---------------------------------------------------------------------------


class CellError:   # stands for xloil.CellError: int() is the COM error code
    def __init__(self, code):
        self.code = code

    def __int__(self):
        return self.code


def _obj(rows):
    return np.array(rows, dtype=object)


@pytest.mark.parametrize("xloil_value, com_value", [
    (None, None),
    (_obj([[2.0]]), 2.0),
    (_obj([[2]]), 2.0),
    (_obj([[None]]), None),
    (_obj([["a"]]), "a"),
    (_obj([[True]]), True),
    (_obj([[CellError(-2146826252)]]), -2146826252),
    (_obj([[1.0], [None], ["x"]]), ((1.0,), (None,), ("x",))),
    (_obj([[1.0, 2.0, 3]]), ((1.0, 2.0, 3.0),)),
    (_obj([[1.0, CellError(-2146826246)], [None, False]]), ((1.0, -2146826246), (None, False))),
])
def test_xloil_values_become_com_values(xloil_value, com_value):
    assert bits(c.xloil_raw(xloil_value)) == bits(com_value)


def test_xloil_result_shows_inf_as_num_error():
    num = object()
    assert c.xloil_result(float("inf"), num) is num
    out = c.xloil_result([[1.0], [float("-inf")], [""]], num)
    assert out.shape == (3, 1) and out[1, 0] is num and out[0, 0] == 1.0 and out[2, 0] == ""
    row = c.xloil_result([1.0, 2.0, 3.0], num)
    assert row.shape == (1, 3)


# ---------------------------------------------------------------------------
# Plots: the same PNG bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_plot", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_plot_png_is_byte_identical(n, log_plot, be_xy, tmp_path):
    xutils = pytest.importorskip("xlwings.utils")
    model = {1: extrapolation._EXPONENTIAL, 2: extrapolation._SQRT_EXPONENTIAL, 3: extrapolation._POWER_LAW}[n]
    x, y = be_xy

    def figure():
        result = extrapolation._extrapolate_quietly(x, y, model, None, None)
        return result.plot_linearized() if log_plot else result.plot()

    theirs, _temp = xutils.process_image(figure(), format="png", export_options=None)
    ours = c.process_image(figure())
    try:
        assert Path(ours).read_bytes() == Path(theirs).read_bytes()
    finally:
        for f in (ours, theirs):
            os.unlink(f)
