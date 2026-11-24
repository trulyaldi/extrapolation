"""
Convert workbooks made with the xlwings add-in (extraplus.xlam) to the xlOil add-in (extraplus.xll).

A formula that used the old add-in refers to its functions through an external link, for example

    ='C:\\Users\\me\\AppData\\Local\\Extraplus\\extraplus.xlam'!EXTRAP1_V($A$2:$A$6,B2:B6)

The xlOil add-in registers the same function names globally, so the conversion removes the link
prefix in front of Extraplus function names and nothing else:

    =EXTRAP1_V($A$2:$A$6,B2:B6)

The arguments, the cells, array formulas and every other formula are left as they are. Once no
formula uses the old link, the link is removed from the workbook (Edit Links). The numbers are
recomputed by the same functions, so they do not change.

convert_formula() is pure; convert_workbook() works on a COM Workbook (Excel's object model).
"""

from __future__ import annotations

import ntpath
import re

FUNCTION_NAMES = (
    [f"{base}{n}" for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE", "FINITE_EXTRAP") for n in (1, 2, 3)]
    + [f"EXTRAP{n}{suffix}" for suffix in ("_V", "_H", "_PLOT", "_PLOT_LOG") for n in (1, 2, 3)]
    + ["EXTRAP_ALL_V", "EXTRAPLUS_VERSION", "EXTRAPLUS_CACHE_STATS"]
)
LEGACY_ADDIN = "extraplus.xlam"
_NAMES = "|".join(sorted(FUNCTION_NAMES, key=len, reverse=True))
# The link prefix as Excel writes it in formula text: a quoted path ('' is a quote inside), or an
# unquoted workbook name/path, ending in extraplus.xlam, then "!" and one of our function names.
_PREFIX = re.compile(
    r"(?:'(?:[^']|'')*?extraplus\.xlam'|(?<![\w.'\]])[^\s'!(),;=+\-*/^&<>{}\"]*?extraplus\.xlam)!"
    rf"(?=(?:{_NAMES})\s*\()",
    re.IGNORECASE)

XL_EXCEL_LINKS = 1
XL_CELL_TYPE_FORMULAS = -4123
XL_CALCULATION_MANUAL = -4135
XL_PART = 2


def convert_formula(text):
    """The formula without the old add-in's link prefixes, or None if it has none."""
    if not isinstance(text, str) or "extraplus.xlam" not in text.lower():
        return None
    new = _PREFIX.sub("", text)
    return None if new == text else new


def prefixes_in(text):
    """The distinct link prefixes in a formula (for Range.Replace)."""
    return {m.group(0) for m in _PREFIX.finditer(text)} if isinstance(text, str) else set()


def legacy_links(workbook):
    """The workbook's links to an extraplus.xlam (any folder)."""
    try:
        links = workbook.LinkSources(XL_EXCEL_LINKS)
    except Exception:
        return []
    if not links:
        return []
    return [str(link) for link in links if ntpath.basename(str(link)).lower() == LEGACY_ADDIN]


def _formula_cells(sheet):
    """(row, column, formula) of every formula cell of the sheet."""
    try:
        found = sheet.UsedRange.SpecialCells(XL_CELL_TYPE_FORMULAS)
    except Exception:   # Excel raises when there are no formula cells
        return
    for area in found.Areas:
        row0, col0, formulas = area.Row, area.Column, area.Formula
        rows = formulas if isinstance(formulas, tuple) else ((formulas,),)
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                yield row0 + i, col0 + j, text


def _remaining(workbook):
    """[(sheet, row, column, formula)] that still use the old add-in's link."""
    left = []
    for sheet in workbook.Worksheets:
        for row, col, text in _formula_cells(sheet):
            if convert_formula(text) is not None:
                left.append((sheet.Name, row, col, text))
    return left


def convert_workbook(workbook) -> dict:
    """
    Remove the extraplus.xlam link prefixes from every formula of a COM workbook, then the link.

    Returns a report: {"links": [...], "cells_before": n, "converted": n, "failed": [(where, why)],
    "links_removed": [...]}.  Calculation, events and screen updating are restored afterwards.
    """
    report = {"links": legacy_links(workbook), "cells_before": 0, "converted": 0, "failed": [],
              "links_removed": []}
    before = _remaining(workbook)
    report["cells_before"] = len(before)
    if not before and not report["links"]:
        return report
    app = workbook.Application
    saved = (app.Calculation, app.EnableEvents, app.ScreenUpdating)
    try:
        app.Calculation = XL_CALCULATION_MANUAL
        app.EnableEvents = False
        app.ScreenUpdating = False
        # 1. Range.Replace per sheet and distinct prefix (fast; also handles array formulas).
        by_sheet = {}
        for sheet_name, _r, _c, text in before:
            by_sheet.setdefault(sheet_name, set()).update(prefixes_in(text))
        for sheet_name, prefixes in by_sheet.items():
            used = workbook.Worksheets(sheet_name).UsedRange
            for prefix in sorted(prefixes, key=len, reverse=True):
                # ~ * ? are Replace wildcards (8.3 paths contain ~): escape them with ~.
                what = prefix.replace("~", "~~").replace("*", "~*").replace("?", "~?")
                try:
                    used.Replace(What=what, Replacement="", LookAt=XL_PART, MatchCase=False)
                except Exception:
                    pass   # the per-cell pass below handles what is left
        # 2. Per cell for anything Replace did not change.
        for sheet_name, row, col, _text in _remaining(workbook):
            cell = workbook.Worksheets(sheet_name).Cells(row, col)
            where = f"'{sheet_name}'!R{row}C{col}"
            try:
                if cell.HasArray:
                    block = cell.CurrentArray
                    new = convert_formula(block.FormulaArray)
                    if new is not None:
                        block.FormulaArray = new
                else:
                    try:
                        text, prop = cell.Formula2, "Formula2"
                    except Exception:   # Excel without dynamic arrays
                        text, prop = cell.Formula, "Formula"
                    new = convert_formula(text)
                    if new is not None:
                        setattr(cell, prop, new)
            except Exception as exc:
                report["failed"].append((where, str(exc)))
        left = _remaining(workbook)
        report["converted"] = report["cells_before"] - len(left)
        for sheet_name, row, col, text in left:
            where = f"'{sheet_name}'!R{row}C{col}"
            if not any(w == where for w, _ in report["failed"]):
                report["failed"].append((where, f"formula still uses the old add-in: {text[:80]}"))
        # 3. The link, once nothing uses it (BreakLink would turn remaining formulas into values).
        if not left:
            for link in legacy_links(workbook):
                try:
                    workbook.BreakLink(Name=link, Type=XL_EXCEL_LINKS)
                    report["links_removed"].append(link)
                except Exception as exc:
                    report["failed"].append((link, f"could not remove the link: {exc}"))
    finally:
        for attr, value in zip(("Calculation", "EnableEvents", "ScreenUpdating"), saved):
            try:
                if getattr(app, attr) != value:
                    setattr(app, attr, value)
            except Exception:
                pass
    return report


def summary(report) -> str:
    lines = [f"Converted {report['converted']} of {report['cells_before']} cells with Extraplus formulas "
             "from the old add-in (extraplus.xlam) to the new one."]
    if report["links_removed"]:
        lines.append("Removed the link to the old add-in.")
    if report["failed"]:
        lines.append(f"{len(report['failed'])} problem(s):")
        lines += [f"  {where}: {why}" for where, why in report["failed"][:15]]
    lines.append("Save the workbook to keep the change.")
    return "\n".join(lines)
