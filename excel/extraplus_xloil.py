"""
Extraplus Excel functions through xlOil (an in-process XLL) instead of xlwings.

The functions are those of extrapolation.py, unchanged: each registered
worksheet function converts its arguments exactly as xlwings 0.37.4 would
(extraplus_xlcompat.read), calls the function object from extrapolation.py,
and hands Excel the result exactly as xlwings writes it
(extraplus_xlcompat.written).  So the names, arguments, help texts, numbers,
"Error: ..." texts, plots and the "Write values" command are the same; only
the bridge to Excel changes.

Differences that remain (the bridge, not the logic):
  - Cells formatted as dates or currency: xlwings (COM Range.Value) reads them
    as datetime / Decimal, xlOil passes the cell's number.  extraplus rejects
    dates, so such inputs gave "Error: ..." under xlwings and are fitted here.
  - expand="table": xlwings re-entered a formula at the size of its result
    when the formula's range had another size.  Here Excel shows the result in
    the formula's cells as it does for any array function (#N/A in extra cells).
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
from pathlib import Path

import numpy as np
import xloil as xlo

import extrapolation as _x
import extraplus_migrate as _m
import extraplus_xlcompat as _c

_ARRAY = xlo.Array(object, dims=2, trim=False)
_CATEGORY = "Extrapolation"
_NUM = xlo.CellError.NUM


def _make(name: str, call):
    """Register `name` with the arguments in _c.SIGNATURES[name]; call(*converted) -> result."""
    spec = _c.SIGNATURES[name]
    options = [opt for _arg, opt, _help, _req in spec]

    def worksheet_function(*args):
        values = [_c.read(_c.xloil_raw(a), opt) for a, opt in zip(args, options)]
        # An argument left out: xlwings passes the Python default (None), not a converted value.
        values = [None if a is None else v for a, v in zip(args, values)]
        return _c.xloil_result(_c.written(call(*values)), _NUM)

    params = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=_ARRAY,
                                default=inspect.Parameter.empty if required else None)
              for arg, _opt, _help, required in spec]
    worksheet_function.__signature__ = inspect.Signature(params)
    worksheet_function.__name__ = name
    original = getattr(_x, name)
    worksheet_function.__doc__ = original.__doc__
    help_text = inspect.cleandoc(original.__doc__ or "").split("\n\n")[0].replace("\n", " ")
    # Error cells are passed to the function as xlwings does (ErrorPropagation=false in extraplus.ini).
    xlo.func(worksheet_function, name=name, help=help_text, group=_CATEGORY,
             args={arg: h for arg, _opt, h, _req in spec if h},
             macro=name.endswith(("_PLOT", "_PLOT_LOG")))
    return worksheet_function


# ---------------------------------------------------------------------------
# Numeric functions: the extrapolation.py functions, called as xlwings calls them
# ---------------------------------------------------------------------------

_VALUE_NAMES = [n for n in _x.__all__ if not n.endswith(("_PLOT", "_PLOT_LOG"))]
for _name in _VALUE_NAMES:
    _make(_name, getattr(_x, _name))


# ---------------------------------------------------------------------------
# Plot functions: extrapolation._run_plot_udf with an xlwings-like caller
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


def _plot_function(name: str, model: str, log_plot: bool):
    def call(x_range, y_range, b_fixed=None, n_fit=None):
        caller = _Caller(xlo.Caller().range.to_com())
        return _x._run_plot_udf(x_range=x_range, y_range=y_range, model_type=model, caller=caller,
                                picture_name=name, log_plot=log_plot, b_fixed=b_fixed, n_fit=n_fit)

    return call


_MODELS = {1: _x._EXPONENTIAL, 2: _x._SQRT_EXPONENTIAL, 3: _x._POWER_LAW}
for _n, _model in _MODELS.items():
    for _suffix, _log in (("_PLOT", False), ("_PLOT_LOG", True)):
        _make(f"EXTRAP{_n}{_suffix}", _plot_function(f"EXTRAP{_n}{_suffix}", _model, _log))


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


xlo.event.WorkbookBeforeSave += _on_before_save


def _check_open_workbooks():
    # Workbooks already open when the add-in loads (opened before it, or with Excel's start).
    try:
        for book in xlo.app().Workbooks:
            _offer_conversion(book.Name)
    except Exception:
        pass


xlo.excel_callback(_check_open_workbooks)

_GUI = xlo.ExcelGUI(ribbon=_RIBBON, funcmap={"writeValues": _on_write_values, "convertWorkbook": _on_convert},
                    name="Extraplus")
