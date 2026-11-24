"""
Extraplus Excel functions through xlOil (an in-process XLL) instead of xlwings.

The worksheet functions below are those of extrapolation.py, copied as they are:
the same names, decorators (func / arg / ret, with xlwings' arguments), bodies,
argument options and docs.  Only the decorators differ: here they register the
function with xlOil, and the registered function

  - reads each argument as xlwings 0.37.4 would (extraplus_xlcompat.read on the
    cell values xlOil passes), so the function gets the same Python values;
  - hands Excel the result as xlwings writes it (extraplus_xlcompat.written);
  - for ret(expand="table") (EXTRAPn_V / _H, EXTRAP_ALL_V), re-enters the formula
    as an array formula of the result's size after the calculation, when the
    formula's cells have another size (xlwings' delayed resize).  Without that,
    an Excel without dynamic arrays shows only the first value of a formula
    typed in one cell.

The helpers (_run_udf, _excel_errors, _vertical_fit_output, _run_plot_udf,
the result cache ...) are imported from extrapolation.py, so both add-ins
compute with the same code.

Without xlOil (tests on Linux), the decorators only record the registration and
the functions stay plain Python functions.

Differences that remain (the bridge, not the logic):
  - Cells formatted as dates or currency: xlwings (COM Range.Value) reads them
    as datetime / Decimal, xlOil passes the cell's number.  extraplus rejects
    dates, so such inputs gave "Error: ..." under xlwings and are fitted here.
  - An exception inside a _PLOT function (they do not catch errors) shows
    xlOil's error text instead of xlwings' traceback text.

Loaded by xlOil through the add-in's ini file (LoadModules = ["extraplus_xloil"]).
"""

from __future__ import annotations

import ctypes
import inspect
import os
import sys
import time
from functools import wraps
from pathlib import Path

import numpy as np

try:
    import xloil as xlo
except ImportError:   # not inside Excel (tests): nothing is registered
    xlo = None

import extrapolation as _x
import extraplus_migrate as _m
import extraplus_xlcompat as _c
from extrapolation import (  # noqa: F401  (used by the functions copied below)
    _CACHE,
    _EXPONENTIAL,
    _POWER_LAW,
    _SQRT_EXPONENTIAL,
    _b_fixed_per_model,
    _error_text,
    _excel_errors,
    _horizontal_fit_output,
    _run_plot_udf,
    _run_udf,
    _vertical_fit_output,
    extraplus,
)

if xlo is not None:
    _ARRAY = xlo.Array(object, dims=2, trim=False)
    _NUM = xlo.CellError.NUM
else:
    _ARRAY = None
    _NUM = "#NUM!"

# name -> {"args": [(name, read options, help, required)], "ret": {...}, "category": ..., "volatile": ...}
REGISTERED = {}


# ---------------------------------------------------------------------------
# func / arg / ret: xlwings' decorators, registering with xlOil
# ---------------------------------------------------------------------------


def arg(name, convert=None, doc=None, **options):
    """xlwings.arg: read options and help of one argument."""
    def decorate(function):
        if convert is not None:
            options["convert"] = convert
        function.__dict__.setdefault("_xlo_args", {})[name] = (dict(options), doc)
        return function
    return decorate


def ret(**options):
    """xlwings.ret: options of the result (expand="table")."""
    def decorate(function):
        function._xlo_ret = dict(options)
        return function
    return decorate


def func(function=None, *, category="", volatile=False):
    """xlwings.func: register the function as a worksheet function."""
    def decorate(function):
        _register(function, category, volatile)
        return function
    return decorate if function is None else decorate(function)


def _register(function, category, volatile):
    name = function.__name__
    declared = function.__dict__.get("_xlo_args", {})
    ret_options = function.__dict__.get("_xlo_ret", {})
    params = [p for p in inspect.signature(function).parameters.values()]
    has_caller = any(p.name == "caller" for p in params)
    spec = []
    for position, p in enumerate((p for p in params if p.name != "caller"), start=1):
        options, doc = declared.get(p.name, ({}, None))
        # Without a doc, xlwings registers "Positional argument <n>" as the argument's help.
        spec.append((p.name, options, doc if doc else f"Positional argument {position}",
                     p.default is inspect.Parameter.empty))
    REGISTERED[name] = {"args": spec, "ret": ret_options, "category": category, "volatile": volatile}
    if xlo is None:
        return

    names = [p.name for p in params]
    expand = ret_options.get("expand")

    def worksheet_function(*args):
        kwargs = {}
        for (arg_name, options, _help, _required), value in zip(spec, args):
            # An argument left out: xlwings passes the Python default (None), not a converted value.
            kwargs[arg_name] = None if value is None else _c.read(_c.xloil_raw(value), options)
        if has_caller:
            kwargs["caller"] = _Caller(xlo.Caller().range.to_com())
        result = function(**{n: kwargs[n] for n in names if n in kwargs})
        if expand:
            _resize_if_needed(result)
        return _c.xloil_result(_c.written(result), _NUM)

    worksheet_function.__signature__ = inspect.Signature(
        [inspect.Parameter(arg_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=_ARRAY,
                           default=inspect.Parameter.empty if required else None)
         for arg_name, _options, _help, required in spec])
    worksheet_function.__name__ = name
    worksheet_function.__doc__ = function.__doc__
    help_text = inspect.cleandoc(function.__doc__ or "").split("\n\n")[0].replace("\n", " ")
    # Error cells reach the function as under xlwings (ErrorPropagation=false in extraplus.ini).
    # Macro type: the plot functions and the resize read the caller's sheet and position, which an
    # ordinary worksheet function may not do.  The arguments are still values (Excel calculates them).
    xlo.func(worksheet_function, name=name, help=help_text, group=category,
             args={arg_name: h for arg_name, _o, h, _r in spec},
             macro=bool(has_caller or expand), volatile=volatile)


# ---------------------------------------------------------------------------
# expand="table": xlwings.udfs.call_udf + delayed_resize_dynamic_array_formula
# ---------------------------------------------------------------------------


def _result_size(result):
    """The result's size as xlwings computes it from the written value (at least 1 x 1)."""
    if isinstance(result, np.ndarray) and result.ndim == 2:
        return (max(1, result.shape[0]), max(1, result.shape[1]))
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
        return (max(1, len(result)), max(1, len(result[0])))
    return (1, 1)


_XL_DONE = 0                 # Application.CalculationState: xlDone
_RESIZE_WAIT_MS = 100        # as xlwings: the resize runs 0.1 s after the call ...
_pending_resizes = set()     # ... once per range, and only when Excel has finished calculating


def _resize_if_needed(result):
    """If the calling cells are not the result's size, re-enter the formula as an array formula of
    that size once the calculation is over (xlwings' expand="table": call_udf schedules
    delayed_resize_dynamic_array_formula). Writing to the sheet is not possible during it."""
    try:
        caller = xlo.Caller().range
        current = (caller.nrows, caller.ncols)
    except Exception:   # not called from a sheet (VBA, Evaluate)
        return
    size = _result_size(result)
    if current == size:
        return
    rng = caller.to_com()   # the calling cells themselves
    address = f"{rng.Worksheet.Name}!{rng.Address}"
    key = (rng.Worksheet.Parent.Name, address)
    if key in _pending_resizes:
        return
    _pending_resizes.add(key)

    def resize():
        try:
            if xlo.app().CalculationState != _XL_DONE:
                xlo.excel_callback(resize, wait=_RESIZE_WAIT_MS)   # still calculating: later
                return
        except Exception:
            pass
        _pending_resizes.discard(key)
        try:
            # The formula's cells as they are now: its whole array formula, or the single cell.
            # (Re-entering a formula recalculates it at once, and during that calculation the caller
            # can be reported as one cell of the new array; the array itself is the truth.)
            first = rng.Cells(1, 1)
            cells = first.CurrentArray if first.HasArray else first
            if (cells.Rows.Count, cells.Columns.Count) == size:
                return
            # xlwings.udfs.delayed_resize_dynamic_array_formula
            formula = cells.FormulaArray
            if not formula:
                formula = cells.Formula
            # Not cells.Resize(rows, cols): through pywin32's dynamic dispatch that is Resize()(rows, cols),
            # i.e. Range.Item(rows, cols), a single cell elsewhere.
            sheet, row, col = first.Worksheet, first.Row, first.Column
            target = sheet.Range(sheet.Cells(row, col), sheet.Cells(row + size[0] - 1, col + size[1] - 1))
            if cells.Columns.Count > size[1] or cells.Rows.Count > size[0]:
                cells.ClearContents()
            target.FormulaArray = formula
        except Exception as exc:   # e.g. the cells below are part of another array formula
            xlo.log.warn(f"Extraplus: could not resize {address} to {size[0]} x {size[1]} cells: {exc}")

    xlo.excel_callback(resize, wait=_RESIZE_WAIT_MS)


# ---------------------------------------------------------------------------
# Plot functions: the caller (xlwings' reserved "caller" argument) and its pictures
# ---------------------------------------------------------------------------


class _Pictures:
    """sheet.pictures.add(fig, name=..., update=True, left=..., top=...) as xlwings does it:
    the same PNG (xlwings.utils.process_image), inserted by the same COM calls.  Writing to the
    sheet is not allowed during calculation, so the insertion runs right after it."""

    def __init__(self, sheet_com):
        self._sheet = sheet_com

    def add(self, fig, name=None, update=False, left=None, top=None):
        filename = _c.process_image(fig)
        sheet = self._sheet

        def insert():
            try:
                existing = None
                if update:
                    try:
                        existing = sheet.Pictures(name)
                    except Exception:
                        existing = None
                if existing is not None:
                    # xlwings.utils.excel_update_picture
                    old = (existing.Left, existing.Top, existing.Width, existing.Height)
                    existing.Delete()
                    shape = sheet.Shapes.AddPicture(Filename=filename, LinkToFile=False, SaveWithDocument=True,
                                                    Left=old[0], Top=old[1], Width=old[2], Height=old[3])
                else:
                    shape = sheet.Shapes.AddPicture(Filename=filename, LinkToFile=False, SaveWithDocument=True,
                                                    Left=left if left else 0, Top=top if top else 0,
                                                    Width=-1, Height=-1)
                shape.DrawingObject.Name = name
            finally:
                try:
                    os.unlink(filename)
                except OSError:
                    pass

        xlo.excel_callback(insert)


class _Sheet:
    def __init__(self, sheet_com):
        self.pictures = _Pictures(sheet_com)


class _Caller:
    """The attributes of xlwings' caller Range that extrapolation._insert_plot_in_excel uses."""

    def __init__(self, range_com):
        self.row = range_com.Row
        self.column = range_com.Column
        self.left = range_com.Left
        self.top = range_com.Top
        self.height = range_com.Height
        self.sheet = _Sheet(range_com.Worksheet)


# ---------------------------------------------------------------------------
# The worksheet functions: extrapolation.py, lines 593-1005 and 1478-1651, as they are
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Extrapolated values at infinity
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).value


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).value


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).value


# ---------------------------------------------------------------------------
# Fitted values at a finite basis size
# ---------------------------------------------------------------------------

# target defaults to None so that Excel calls Python (and shows "Error: target
# is required.") when it is left out, instead of rejecting the formula.


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP1(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Exponential fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).predict(target)


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP2(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Square-root-exponential fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).predict(target)


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP3(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Power-law fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).predict(target)


# ---------------------------------------------------------------------------
# Uncertainty values
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential B for the normalized variable x/x_max."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).decay_rate


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential B for sqrt(x/x_max)."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).decay_rate


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law B for (x/x_max)^(-B)."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).decay_rate


# ---------------------------------------------------------------------------
# Combined spill functions: C, uncertainty, B
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
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


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
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


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
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


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
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


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
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


@func(category="Extrapolation", volatile=False)
@ret(expand="table")
@arg("x_range", np.array, ndim=2)
@arg("y_range", np.array, ndim=2)
@arg("b_fixed", err_to_str=True)
@arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
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
# All three models in one call
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", ndim=2, err_to_str=True,   # not np.array: it reads blank cells as NaN
        doc="Optional: 3 cells with a fixed B for models 1, 2, 3 (a blank cell fits that B). Leave out to fit all.")
@_excel_errors(shape=(3, 3))
def EXTRAP_ALL_V(x_range, y_range, b_fixed=None):
    """All three models: columns 1-3 are EXTRAP1_V-EXTRAP3_V (rows C, uncertainty, B)."""
    columns = []
    for model, b in zip((_EXPONENTIAL, _SQRT_EXPONENTIAL, _POWER_LAW), _b_fixed_per_model(b_fixed)):
        try:
            columns.append(_vertical_fit_output(_run_udf(x_range, y_range, model, b)))
        except Exception as exc:   # this model only: the others are still shown
            columns.append(np.full((3, 1), _error_text(exc), dtype=object))
    return np.hstack(columns)


# ---------------------------------------------------------------------------
# Installed version
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@_excel_errors
def EXTRAPLUS_VERSION():
    """Installed version of the extraplus package."""
    return extraplus.__version__


@func(category="Extrapolation", volatile=False)
@_excel_errors
def EXTRAPLUS_CACHE_STATS():
    """State of the result cache: mode, file, hits and misses (Ctrl+Alt+F9 refreshes it)."""
    return _CACHE.stats()



# ---------------------------------------------------------------------------
# Matplotlib plots in Excel
# ---------------------------------------------------------------------------


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


@func(category="Extrapolation", volatile=False)
@arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
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


# ---------------------------------------------------------------------------
# Write values: extrapolation._values_pass on a COM workbook adapter
# ---------------------------------------------------------------------------


class _ComBook(_x._XlwingsBook):
    """_XlwingsBook with xlwings' Range reading and writing replaced by the same steps on COM."""

    def __init__(self, book_com, scope_com):
        self.book_com, self.app_com, self.scope_com = book_com, book_com.Application, scope_com

    def formula_cells(self):
        name = self.scope_com.Worksheet.Name
        try:
            found = self.scope_com.SpecialCells(_x._XL_CELL_TYPE_FORMULAS)
        except Exception:   # Excel raises when there are no formula cells
            return
        for area in found.Areas:
            row0, col0, formulas = area.Row, area.Column, area.Formula
            rows = formulas if isinstance(formulas, tuple) else ((formulas,),)
            for i, row in enumerate(rows):
                for j, text in enumerate(row):
                    if isinstance(text, str) and _x._MENTIONS_RE.search(text):
                        yield name, row0 + i, col0 + j, text

    def block_of(self, sheet, row, col):
        cell = self.book_com.Worksheets(sheet).Cells(row, col)
        block = None
        if cell.HasArray:
            block = cell.CurrentArray
        else:
            try:
                if cell.HasSpill:
                    block = cell.SpillingToRange
            except Exception:   # no dynamic arrays in this Excel
                pass
        if block is None:
            return (row, col, row, col)
        return (block.Row, block.Column, block.Row + block.Rows.Count - 1, block.Column + block.Columns.Count - 1)

    def _range(self, sheet, box):
        ws = self.book_com.Worksheets(sheet)
        return ws.Range(ws.Cells(box[0], box[1]), ws.Cells(box[2], box[3]))

    def read(self, sheet, area, options):
        return _c.read(self.book_com.Worksheets(sheet).Range(area).Value, options)

    def evaluate(self, text):
        return self.app_com.Evaluate(text)

    def convert(self, value, options):
        return _c.read(value, options)

    def values(self, sheet, box):
        return _c.read(self._range(sheet, box).Value, {"ndim": 2})

    def write(self, sheet, box, result):
        data = _c.written(result)
        data = data if isinstance(data, list) and data and isinstance(data[0], list) else [
            data if isinstance(data, list) else [data]]
        target = self._range(sheet, box)
        try:
            target.Value = data
        except Exception:   # e.g. Excel refuses to overwrite an array formula cell by cell
            target.ClearContents()
            target.Value = data

    def backup(self):
        full = self.book_com.FullName
        if not self.book_com.Path or "://" in full:
            raise RuntimeError("the workbook must be saved in a local folder first (a backup copy is made next to it)")
        path = Path(full)
        stamp = time.strftime("%Y-%m-%d %H%M%S")
        copy = path.with_name(f"{path.stem} (before Extraplus values {stamp}){path.suffix}")
        self.book_com.SaveCopyAs(str(copy))
        return str(copy)

    def calculation_paused(self):
        app = self.app_com

        class _Paused:
            def __enter__(inner):
                inner.saved = (app.Calculation, app.EnableEvents, app.ScreenUpdating)
                app.Calculation = _x._XL_CALCULATION_MANUAL
                app.EnableEvents = False
                app.ScreenUpdating = False

            def __exit__(inner, *exc):
                for attr, value in zip(("Calculation", "EnableEvents", "ScreenUpdating"), inner.saved):
                    try:
                        if getattr(app, attr) != value:
                            setattr(app, attr, value)
                    except Exception:
                        pass
                return False

        return _Paused()

    def record(self, records, report):
        sheets = self.book_com.Worksheets
        names = [sheets(i).Name for i in range(1, sheets.Count + 1)]
        if _x._RECORD_SHEET in names:
            sheet = sheets(_x._RECORD_SHEET)
        else:
            sheet = sheets.Add(After=sheets(sheets.Count))
            sheet.Name = _x._RECORD_SHEET
            sheet.Range("A1:F1").Value = [["Time", "Range", "Formula (as it was)", "Function",
                                           "Inputs SHA-256", "Values written / note"]]
        used = sheet.UsedRange
        next_row = used.Row + used.Rows.Count
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fingerprint = _x._code_fingerprint().hex()[:12]
        rows = [["run", "", f"extraplus {_x.extraplus.__version__} ({fingerprint})", "",
                 _x._summary({**report, "problems": []}).replace("\n", " ")]]
        rows += [[where, formula, name, digest, written] for where, formula, name, digest, written in records]
        rows += [[where, "", status, "", why] for where, status, why in report["problems"]]
        data = [[stamp] + ["'" + str(v) if v else "" for v in row] for row in rows]
        sheet.Range(sheet.Cells(next_row, 1), sheet.Cells(next_row + len(data) - 1, len(data[0]))).Value = data


_MB_OK, _MB_OKCANCEL, _MB_DEFBUTTON2 = 0x0, 0x1, 0x100
_MB_ICONERROR, _MB_ICONQUESTION, _MB_ICONINFO = 0x10, 0x20, 0x40
_IDOK = 1


def _message(text, title, flags):
    return ctypes.windll.user32.MessageBoxW(xlo.app().Hwnd, text, title, flags)


def write_values() -> None:
    """The ribbon command: the add-in's ExtraplusWriteValues macro and extrapolation.write_values in one."""
    app = xlo.app()
    book = app.ActiveWorkbook
    if book is None:
        return
    # As the add-in's VBA did: the selection counts only if it is a Range of more than one cell.
    selection = app.Selection
    scope = ""
    try:
        if selection is not None and selection.CountLarge > 1:
            scope = "the selected cells"
    except Exception:   # not a Range (a shape, a chart ...)
        selection = None
    if scope == "":
        scope = f"the sheet '{app.ActiveSheet.Name}'"
    if book.Path == "":
        _message("Save the workbook first: a backup copy is saved next to it before any formula is replaced.",
                 "Extraplus", _MB_OK | _MB_ICONINFO)
        return
    if _message(f"Replace the Extraplus formulas in {scope} with the numbers they compute?\n\n"
                "The numbers are computed again from the formulas' input cells. Before anything is replaced, "
                "a copy of the workbook is saved next to it, and each replaced formula is listed on the "
                "sheet 'Extraplus values'. Formulas that give an error are kept.",
                "Extraplus", _MB_OKCANCEL | _MB_ICONQUESTION | _MB_DEFBUTTON2) != _IDOK:
        return
    title = "Extraplus: write values"
    try:
        target = selection if scope == "the selected cells" else app.ActiveSheet.UsedRange
        report = _x._values_pass(_ComBook(book, target))
    except Exception as exc:
        _message(f"Nothing was written: {_x._error_text(exc)}", title, _MB_OK | _MB_ICONERROR)
        return
    _message(_x._summary(report), title, _MB_OK | (_MB_ICONERROR if report["stopped"] else _MB_ICONINFO))


_RIBBON = '''<customUI xmlns="http://schemas.microsoft.com/office/2009/07/customui">
  <ribbon>
    <tabs>
      <tab id="extraplusTab" label="Extraplus">
        <group id="extraplusResults" label="Results">
          <button id="extraplusWriteValues" label="Write values" size="large" imageMso="PasteValues"
                  onAction="writeValues"
                  screentip="Replace Extraplus formulas by their values"
                  supertip="Computes the Extraplus formulas in the selection (or on the sheet) again and replaces them with the numbers, so the sheet no longer recalculates them. A backup copy of the workbook is saved first."/>
          <button id="extraplusConvert" label="Convert old formulas" size="large" imageMso="Refresh"
                  onAction="convertWorkbook"
                  screentip="Use the new add-in in this workbook"
                  supertip="Formulas made with the earlier Extraplus add-in (extraplus.xlam) refer to it by its file. This removes that reference so the formulas use the installed add-in. The formulas and their numbers stay the same; save the workbook afterwards."/>
        </group>
      </tab>
    </tabs>
  </ribbon>
</customUI>'''


def _on_write_values(ctrl):
    write_values()


# ---------------------------------------------------------------------------
# Workbooks made with the xlwings add-in (extraplus.xlam)
# ---------------------------------------------------------------------------

_TITLE_CONVERT = "Extraplus: convert old formulas"
_MB_YESNO, _IDYES = 0x4, 6


def convert_workbook(book, ask: bool) -> None:
    """Convert a workbook from extraplus.xlam to this add-in (extraplus_migrate), asking first if ask."""
    links = _m.legacy_links(book)
    if not links and not ask:
        return
    if ask:
        if not links:
            _message("This workbook does not use the earlier Extraplus add-in (extraplus.xlam).",
                     _TITLE_CONVERT, _MB_OK | _MB_ICONINFO)
            return
    report = _m.convert_workbook(book)
    _message(_m.summary(report), _TITLE_CONVERT,
             _MB_OK | (_MB_ICONERROR if report["failed"] else _MB_ICONINFO))


def _offer_conversion(name: str) -> None:
    """When a workbook that uses extraplus.xlam opens in an interactive Excel, offer to convert it."""
    app = xlo.app()
    try:
        if not (app.Visible and app.UserControl):
            return   # Excel driven by a program: never show a dialog
        book = app.Workbooks(name)
    except Exception:
        return
    links = _m.legacy_links(book)
    if not links:
        return
    answer = _message(
        f"'{name}' was made with the earlier Extraplus add-in (extraplus.xlam), which is no longer "
        "installed.\n\nConvert its Extraplus formulas to the installed add-in now? The formulas and "
        "their numbers stay the same; save the workbook afterwards to keep the change.\n\n"
        "(You can also do this later: Extraplus tab > Convert old formulas.)",
        _TITLE_CONVERT, _MB_YESNO | _MB_ICONQUESTION)
    if answer == _IDYES:
        convert_workbook(book, ask=False)


def _on_workbook_open(*args):
    name = args[-1] if args else None
    if name:
        xlo.excel_callback(lambda: _offer_conversion(name))


def _on_convert(ctrl):
    book = xlo.app().ActiveWorkbook
    if book is not None:
        convert_workbook(book, ask=True)


def ExtraplusConvertWorkbook():
    """Command (Application.Run "ExtraplusConvertWorkbook"): convert the active workbook without dialogs."""
    book = xlo.app().ActiveWorkbook
    if book is not None:
        report = _m.convert_workbook(book)
        xlo.app().StatusBar = _m.summary(report).replace("\n", " ")[:250]


if xlo is not None:
    xlo.func(ExtraplusConvertWorkbook, command=True)
    xlo.event.WorkbookOpen += _on_workbook_open


# ---------------------------------------------------------------------------
# Save guard (the add-in's ConfirmSave / CountBrokenCells VBA, unchanged in behaviour)
# ---------------------------------------------------------------------------

_GUARD_MIN_ERRORS = 1           # warn before saving when at least this many Extraplus cells show an error
_XL_ERRORS = 16                 # xlErrors (SpecialCells value type)
_BROKEN = {-2146826259, -2146826265, -2146826273}   # #NAME?, #REF!, #VALUE! (COM error codes)


def _is_extraplus_formula(formula: str) -> bool:
    formula = formula.upper()
    return "EXTRAP" in formula or "UNCERTAINTY" in formula or "DECAY_RATE" in formula


def _count_broken_cells(book):
    count, example = 0, ""
    for ws in book.Worksheets:
        try:
            found = ws.UsedRange.SpecialCells(_x._XL_CELL_TYPE_FORMULAS, _XL_ERRORS)
        except Exception:   # no error cells, or a protected sheet
            continue
        for area in found.Areas:
            for cell in area.Cells:
                if cell.Value2 in _BROKEN and _is_extraplus_formula(cell.Formula):
                    count += 1
                    if not example:
                        example = f"'{ws.Name}'!{cell.Address.replace('$', '')}"
    return count, example


def _confirm_save(book) -> bool:
    """False only when the user, warned, chose not to save. Any problem lets the save go ahead."""
    try:
        app = xlo.app()
        if book.IsAddin or not app.DisplayAlerts:
            return True
        try:
            if book.AutoSaveOn:
                return True
        except Exception:   # the property does not exist in every Excel
            pass
        n, example = _count_broken_cells(book)
        if n < _GUARD_MIN_ERRORS:
            return True
        return _message(
            f"{n} cells with Extraplus formulas show #NAME?, #REF! or #VALUE! (for example {example}).\n\n"
            "This usually means that the Extraplus add-in was not found or the calculation failed. "
            "Saving now replaces the numbers stored in the file with these errors.\n\nSave anyway?",
            "Extraplus", _MB_YESNO | 0x30 | _MB_DEFBUTTON2) == _IDYES   # 0x30 = exclamation icon
    except Exception:
        return True


def _on_before_save(name, save_as_ui, cancel):
    try:
        book = xlo.app().Workbooks(name)
    except Exception:
        return
    if not _confirm_save(book):
        cancel.value = True


if xlo is not None:
    xlo.event.WorkbookBeforeSave += _on_before_save


def _check_open_workbooks():
    # Workbooks already open when the add-in loads (opened before it, or with Excel's start).
    try:
        for book in xlo.app().Workbooks:
            _offer_conversion(book.Name)
    except Exception:
        pass


if xlo is not None:
    xlo.excel_callback(_check_open_workbooks)

if xlo is not None:
    _GUI = xlo.ExcelGUI(ribbon=_RIBBON, funcmap={"writeValues": _on_write_values, "convertWorkbook": _on_convert},
                        name="Extraplus")
