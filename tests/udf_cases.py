"""
Fixed inputs for the Excel-function tests, and the exact bits of every numeric
UDF output for them.  Shared by the in-process tests and the subprocesses
started by test_cache.py, so it imports nothing from the tests package.
"""

import struct

import numpy as np

from conftest import BE_1SE_X, BE_1SE_Y

# Li 2Se energy, first 8 basis sizes of research/data/large-dataset/Li_2Se.csv.
LI_2SE_X = [200, 300, 400, 500, 600, 700, 800, 900]
LI_2SE_Y = [
    -7.4779120726761, -7.47805720308475, -7.47805940348679, -7.47805996971751,
    -7.47806006807216, -7.47806015637655, -7.47806021328825, -7.47806025331565,
]

NAN = float("nan")

# name: (x cells, y cells, b_fixed, n_fit).  Cells as Excel gives them: floats,
# None for blank cells, text.
DATASETS = {
    "be_12": (BE_1SE_X, BE_1SE_Y, None, None),
    "be_3": (BE_1SE_X[:3], BE_1SE_Y[:3], None, None),         # sigma = inf
    "be_4": (BE_1SE_X[:4], BE_1SE_Y[:4], None, None),
    "be_5": (BE_1SE_X[:5], BE_1SE_Y[:5], None, None),
    "be_b_fixed": (BE_1SE_X, BE_1SE_Y, 5.0, None),
    "be_n_fit": (BE_1SE_X, BE_1SE_Y, None, 6),
    "be_both": (BE_1SE_X, BE_1SE_Y, 4.0, 7.9),
    "be_padded": (["Basis Size", *BE_1SE_X, None, ""], ["Energy", *BE_1SE_Y, None, 1.0], None, None),
    "be_rising": (BE_1SE_X, [-v for v in BE_1SE_Y], None, None),
    "li_8": (LI_2SE_X, LI_2SE_Y, None, None),
    "li_blank_b": (LI_2SE_X, LI_2SE_Y, "", ""),
    "nan_cell": ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 2.0, NAN, 1.2, 1.1], None, None),
    "negative_zero": ([1.0, 2.0, 3.0, 4.0], [-0.0, -1.0, -1.5, -1.7], None, None),
    "nan_result": ([1.0, 2.0, 3.0, 4.0], [1.7e308, -1.7e308, 1e308, 0.0], None, None),   # C = nan
}

# Inputs every model rejects with a ValueError ("Error: ..." in Excel).
FAILING = {
    "two_points": ([100, 200], [1.0, 2.0], None, None),
    "constant_y": ([100, 200, 300], [1.0, 1.0, 1.0], None, None),
    "bad_b_fixed": ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], -1.0, None),
    "text_n_fit": ([100, 200, 300, 400], [3.0, 2.0, 1.5, 1.2], None, "abc"),
}

MODELS = (1, 2, 3)
FINITE_TARGETS = (15000, 1300.0)


def column(values):
    """Shape a list like an Excel column range passed with np.array, ndim=2."""
    return np.asarray(values, dtype=object).reshape(-1, 1)


def bits(value):
    """An exact, comparable form of a UDF return value (NaN payloads and -0.0 included)."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("object", value.shape, tuple(bits(v) for v in value.ravel()))
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    return (type(value).__name__, value)


def udf_outputs(extrapolation, datasets=DATASETS):
    """Every numeric UDF's output for every dataset, as {label: bits}."""
    out = {}
    for name, (x, y, b_fixed, n_fit) in datasets.items():
        X, Y = column(x), column(y)
        for n in MODELS:
            for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE"):
                out[f"{name}/{base}{n}"] = bits(getattr(extrapolation, f"{base}{n}")(X, Y, b_fixed, n_fit))
            for suffix in ("_V", "_H"):
                out[f"{name}/EXTRAP{n}{suffix}"] = bits(
                    getattr(extrapolation, f"EXTRAP{n}{suffix}")(X, Y, b_fixed, n_fit))
            for target in FINITE_TARGETS:
                out[f"{name}/FINITE_EXTRAP{n}@{target}"] = bits(
                    getattr(extrapolation, f"FINITE_EXTRAP{n}")(X, Y, target, b_fixed, n_fit))
    return out
