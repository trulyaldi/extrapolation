"""
xlwings-compatible argument and return conversion, without xlwings.

The xlOil add-in (extraplus_xloil.py) calls the unchanged functions of
extrapolation.py.  For their results to be the same as under xlwings, each
function must receive exactly the Python values xlwings would give it, and its
result must reach the cells exactly as xlwings writes it.  This module
reproduces those two steps of xlwings 0.37.4 (the pinned version), stage by
stage:

  reading an argument  (xlwings.conversion ValueAccessor.reader):
      Ensure2DStage -> CleanDataFromReadStage (_xlwindows._clean_value_data_element)
      -> AdjustDimensionsStage(ndim) -> converter (np.array: NumpyArrayConverter)
  writing a result     (xlwings.conversion write + com_server.CallUDF):
      np.ndarray.tolist() -> Ensure2DStage -> CleanDataForWriteStage
      (_xlwindows.prepare_xl_data_element) -> CallUDF squeezing of 1 x n results

The input is the value as COM's Range.Value gives it: a scalar for one cell, a
tuple of row tuples otherwise, numbers as float, blank cells as None, errors
as their COM integer codes, date-formatted cells as pywintypes datetimes.
xloil_raw() turns what xlOil passes into that form.  tests/test_xloil_compat.py
checks every function here against xlwings itself.

No xlOil and no Excel are needed to import this module.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import uuid
from typing import Any

import numpy as np

try:  # pandas is an extraplus dependency; xlwings uses pd.isna when pandas is present
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:  # Windows: the COM date type (a datetime subclass)
    import pywintypes

    _TIME_TYPES = (dt.date, dt.datetime, pywintypes.TimeType, np.datetime64)
except ImportError:
    pywintypes = None
    _TIME_TYPES = (dt.date, dt.datetime, np.datetime64)

# xlwings._xlwindows.cell_errors: COM error codes xlwings turns into text (err_to_str=True)
# or None.  Other error codes (#SPILL!, #CALC!, ...) pass through as ints, as in xlwings.
CELL_ERRORS = {
    -2146826281: "#DIV/0!",
    -2146826246: "#N/A",
    -2146826259: "#NAME?",
    -2146826288: "#NULL!",
    -2146826252: "#NUM!",
    -2146826265: "#REF!",
    -2146826273: "#VALUE!",
}

# Read options of the arguments, as registered with xlwings in extrapolation.py.
RANGE = {"convert": np.array, "ndim": 2}
ERRORS_AS_TEXT = {"err_to_str": True}
PLAIN = {}


# ---------------------------------------------------------------------------
# Reading arguments
# ---------------------------------------------------------------------------


def _com_time_to_datetime(com_time):
    # xlwings._xlwindows._com_time_to_datetime with the default builder (datetime.datetime)
    return dt.datetime(month=com_time.month, day=com_time.day, year=com_time.year, hour=com_time.hour,
                       minute=com_time.minute, second=com_time.second, microsecond=com_time.microsecond,
                       tzinfo=None)


def _clean_element(value, empty_as, err_to_str):
    # xlwings._xlwindows._clean_value_data_element (dates=datetime, numbers=None)
    if value in ("", None):
        return empty_as
    elif isinstance(value, _TIME_TYPES):
        return _com_time_to_datetime(value)
    elif isinstance(value, int) and value in CELL_ERRORS:
        if err_to_str:
            return CELL_ERRORS[value]
        else:
            return None
    return value


def _ensure_2d(value):
    # xlwings.conversion.standard.Ensure2DStage
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            if not isinstance(value[0], (list, tuple)):
                value = [value]
        return value
    return [[value]]


def _adjust_dimensions(value, ndim):
    # xlwings.conversion.standard.AdjustDimensionsStage
    if ndim in (None, "squeeze"):
        if len(value) == 1:
            return value[0][0] if len(value[0]) == 1 else value[0]
        elif len(value[0]) == 1:
            return [x[0] for x in value]
        return value
    elif ndim == 1:
        if len(value) == 1:
            return value[0]
        elif len(value[0]) == 1:
            return [x[0] for x in value]
        raise Exception("Range must be 1-by-n or n-by-1 when ndim=1.")
    elif ndim == "natural":
        if len(value) == 1 and len(value[0]) == 1:
            return value[0][0]
        elif len(value) == 1:
            return value[0]
        return value
    elif ndim != 2:
        raise ValueError("Invalid c.value ndim=%s" % ndim)
    return value


def read(raw: Any, options: dict) -> Any:
    """What xlwings' conversion.read(range_or_None, raw, options) returns for this raw COM value."""
    convert = options.get("convert", None)
    if convert is not None and convert is not np.array:
        raise NotImplementedError(f"converter {convert!r}")
    # NumpyArrayConverter.base_reader: Options(options).defaults(empty=np.nan)
    empty_as = options.get("empty", np.nan if convert is np.array else None)
    err_to_str = options.get("err_to_str", False)
    value = _ensure_2d(raw)
    value = [[_clean_element(c, empty_as, err_to_str) for c in row] for row in value]
    value = _adjust_dimensions(value, options.get("ndim", None))
    if convert is np.array:
        # NumpyArrayConverter.read_value
        ndim = options.get("ndim", None) or 0
        return np.array(value, dtype=options.get("dtype", None), copy=options.get("copy", True),
                        order=options.get("order", None), ndmin=ndim)
    return value


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------


def _prepare_element(x):
    # xlwings._xlwindows.Engine.prepare_xl_data_element (dates are never returned here)
    if isinstance(x, _TIME_TYPES):
        raise NotImplementedError("date results")
    elif pd is not None and pd.isna(x):
        return ""
    elif isinstance(x, (np.floating, float)) and np.isnan(x):
        return ""
    elif isinstance(x, np.number):
        return float(x)
    elif x is None:
        return ""
    return x


def written(ret: Any) -> Any:
    """What xlwings hands back to Excel for a UDF result (CallUDF's return value):
    a scalar, a list (one row) or a list of row lists, cleaned as xlwings cleans them."""
    if isinstance(ret, np.ndarray):
        value = ret.tolist()            # NumpyArrayConverter.write_value
    else:
        value = ret
    value = _ensure_2d(value)
    value = [[_prepare_element(x) for x in row] for row in value]
    # xlwings.com_server.CallUDF
    if len(value) == 1 and len(value[0]) == 1:
        return value[0][0]
    if len(value) == 1 and len(value[0]) > 1:
        return value[0]
    return value


# ---------------------------------------------------------------------------
# From xlOil values to COM values
# ---------------------------------------------------------------------------


def xloil_raw(value: Any) -> Any:
    """
    An argument as xlOil passes it, with xlo.Array(object, dims=2, trim=False) (a 2-d object
    array; missing arguments are None), in the form COM's Range.Value has: one cell is a
    scalar, more cells a tuple of row tuples; numbers are floats, errors their COM codes.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape == (1, 1):
            return _xloil_element(value[0, 0])
        if value.ndim == 1:
            value = value.reshape(1, -1)
        return tuple(tuple(_xloil_element(v) for v in row) for row in value)
    return _xloil_element(value)


def _xloil_element(v):
    if v is None or isinstance(v, (str, bool, float)):
        return v
    if isinstance(v, int):
        return float(v)              # Excel numbers are doubles; COM gives float
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if type(v).__name__ == "CellError":
        return int(v)                # xlOil: "the integer value corresponds to its VBA/COM error number"
    return v


def xloil_result(value: Any, num_error: Any) -> Any:
    """written(ret) in the form xlOil writes to cells.  Excel cannot hold +-inf: a COM
    (xlwings) result of +-inf shows as #NUM!, so it is given as that error here."""
    def cell(x):
        if isinstance(x, float) and np.isinf(x):
            return num_error
        return x

    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return np.array([[cell(x) for x in row] for row in value], dtype=object)
        return np.array([[cell(x) for x in value]], dtype=object)
    return cell(value)


# ---------------------------------------------------------------------------
# Function signatures (the same arguments, options and help as extrapolation.py)
# ---------------------------------------------------------------------------

_HELP_X = "Range containing basis-size x values."
_HELP_X_POS = "Range containing positive basis-size x values."
_HELP_Y = "Range containing the corresponding y values."
_HELP_B = "Optional positive fixed B. Leave blank to fit B."
_HELP_N = "Optional number of first numeric rows used in the fit."
_HELP_T = "Basis size to evaluate the fit at, usually beyond the data (a number > 0)."
_HELP_B_ALL = ("Optional: 3 cells with a fixed B for models 1, 2, 3 (a blank cell fits that B). "
               "Leave out to fit all.")


def _standard(x_help=_HELP_X, docs=True):
    # Without a doc, xlwings registers "Positional argument <n>" as the argument's help.
    helps = [x_help, _HELP_Y, _HELP_B, _HELP_N] if docs else [f"Positional argument {i}" for i in (1, 2, 3, 4)]
    return [("x_range", RANGE, helps[0], True), ("y_range", RANGE, helps[1], True),
            ("b_fixed", ERRORS_AS_TEXT, helps[2], False), ("n_fit", ERRORS_AS_TEXT, helps[3], False)]


def _finite(x_help):
    s = _standard(x_help)
    return s[:2] + [("target", PLAIN, _HELP_T, False)] + s[2:]


def _plot(x_help):
    # The plot functions take b_fixed and n_fit without err_to_str (as registered with xlwings).
    return [("x_range", RANGE, x_help, True), ("y_range", RANGE, _HELP_Y, True),
            ("b_fixed", PLAIN, _HELP_B, False), ("n_fit", PLAIN, _HELP_N, False)]


# name -> [(argument, read options, help, required)]
SIGNATURES = {}
for _n in (1, 2, 3):
    _x = _HELP_X_POS if _n == 3 else _HELP_X
    SIGNATURES[f"EXTRAP{_n}"] = _standard(_x)
    SIGNATURES[f"FINITE_EXTRAP{_n}"] = _finite(_x)
    for _base in ("UNCERTAINTY", "DECAY_RATE"):
        SIGNATURES[f"{_base}{_n}"] = _standard(docs=False)
    for _suffix in ("_V", "_H"):
        SIGNATURES[f"EXTRAP{_n}{_suffix}"] = _standard(docs=False)
    SIGNATURES[f"EXTRAP{_n}_PLOT"] = _plot(_x)
    SIGNATURES[f"EXTRAP{_n}_PLOT_LOG"] = _plot(_x)
SIGNATURES["EXTRAP_ALL_V"] = [("x_range", RANGE, _HELP_X, True), ("y_range", RANGE, _HELP_Y, True),
                              ("b_fixed", {"ndim": 2, **ERRORS_AS_TEXT}, _HELP_B_ALL, False)]
SIGNATURES["EXTRAPLUS_VERSION"] = []
SIGNATURES["EXTRAPLUS_CACHE_STATS"] = []


# ---------------------------------------------------------------------------
# Plot images
# ---------------------------------------------------------------------------


def process_image(fig) -> str:
    """xlwings.utils.process_image for a Matplotlib figure with format="png" and the default
    export options: the PNG file xlwings inserts.  Returns the temporary file name."""
    import matplotlib as mpl
    import matplotlib.backends.backend_agg  # noqa: F401
    import matplotlib.pyplot as plt

    export_options = {"bbox_inches": "tight", "dpi": 200}
    temp_dir = os.path.realpath(tempfile.gettempdir())
    filename = os.path.join(temp_dir, str(uuid.uuid4()) + ".png")
    canvas = mpl.backends.backend_agg.FigureCanvas(fig)
    canvas.draw()
    fig.savefig(filename, **export_options)
    plt.close(fig)
    return filename
