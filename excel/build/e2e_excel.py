"""
End-to-end test of an installed Extraplus in a real Excel (Windows only).

Run with the INSTALLED environment's Python, so the reference values come from
the same packages the add-in uses:

    %EXTRAPLUS_HOME%\\venv\\Scripts\\python.exe e2e_excel.py --book B.xlsx --results R.json [options]

Excel is started the way a user starts it, by running EXCEL.EXE on --book (a COM
automation instance would not load add-ins), and then attached to. It runs as a
child of this process:

    --launch inherit        with this process's environment unchanged
    --launch without-home   without EXTRAPLUS_HOME, to exercise the add-in's
                            registry fallback

Whether an Excel started from the Windows shell picks up EXTRAPLUS_HOME is not
tested here; that is checked by hand.

--mode new (default) creates --book if needed, writes the test data and
formulas, compares every cell with extraplus called directly, and saves the
book. --mode reopen only recalculates an existing book and compares.
--mode startup only opens --book, records the dialog boxes Excel shows and
quits (for example after uninstalling: no "cannot find extraplus.xlam"); it
does not need an Extraplus install.

Formulas are written in two labelled steps, each write logged (cell, formula,
property, outcome) after waiting for Application.Ready:

    single-cell  functions returning one value, written with Range.Formula
    spill        EXTRAP1_V / EXTRAP1_H (xlwings expand="table"), also written with
                 Range.Formula: on Excel 2021 16.0.14332 every Range.Formula2
                 write and HasSpill read fails with 0x800A03EC, even for =1+1.
                 xlwings then fills the 3 cells as a legacy array formula, so
                 this checks that output; dynamic-array spill is not tested
                 via COM.

A failure in the spill step is recorded as its own result and does not hide the
single-cell results.
"""

import argparse
import ctypes
import ctypes.wintypes
import json
import math
import os
import subprocess
import sys
import time
import traceback
import winreg
from pathlib import Path

import pywintypes
import xlwings as xw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_reference import DATA, MODELS, reference  # noqa: E402

N = len(DATA)
X = f"A2:A{N + 1}"
Y = f"B2:B{N + 1}"
CORE = {
    "D2": f"=EXTRAP1({X},{Y})", "D3": f"=EXTRAP2({X},{Y})", "D4": f"=EXTRAP3({X},{Y})",
    "D5": f"=UNCERTAINTY1({X},{Y})", "D6": f"=UNCERTAINTY2({X},{Y})", "D7": f"=UNCERTAINTY3({X},{Y})",
    "D8": f"=DECAY_RATE1({X},{Y})", "D9": f"=DECAY_RATE2({X},{Y})", "D10": f"=DECAY_RATE3({X},{Y})",
    "D11": "=EXTRAPLUS_VERSION()",
    "D12": "=EXTRAP1(A2:A3,B2:B3)",   # too few points -> "Error: ..."
}
# [C, uncertainty, B] of EXTRAP1, down F2:F4 and across H2:J2.
SPILL = {"F2": (f"=EXTRAP1_V({X},{Y})", ["F2", "F3", "F4"]),
         "H2": (f"=EXTRAP1_H({X},{Y})", ["H2", "I2", "J2"])}

# COM errors that mean "Excel is busy, try again", not "this call is wrong".
BUSY_HRESULTS = {
    -2147418111,  # RPC_E_CALL_REJECTED
    -2147417846,  # RPC_E_SERVERCALL_RETRYLATER
    -2146777998,  # VBA_E_IGNORE (0x800AC472)
}


def excel_exe():
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\excel.exe") as key:
                return winreg.QueryValue(key, None)
        except OSError:
            pass
    raise SystemExit("EXCEL.EXE not found")


def retry(fn, timeout=90, what="Excel"):
    deadline = time.time() + timeout
    while True:
        try:
            return fn()
        except Exception as exc:  # Excel rejects calls while it starts or calculates
            if time.time() > deadline:
                raise RuntimeError(f"timed out waiting for {what}: {exc!r}") from exc
            time.sleep(1)


def is_busy(exc):
    if not isinstance(exc, pywintypes.com_error):
        return False
    hresult, _, excepinfo, _ = exc.args
    return hresult in BUSY_HRESULTS or bool(excepinfo and excepinfo[5] in BUSY_HRESULTS)


def wait_ready(app, timeout=60):
    """Seconds waited until Application.Ready is True."""
    started = time.time()
    while True:
        try:
            if app.api.Ready:
                return round(time.time() - started, 2)
        except Exception as exc:
            if not is_busy(exc):
                raise
        if time.time() - started > timeout:
            raise RuntimeError(f"Application.Ready still False after {timeout} s")
        time.sleep(0.25)


class WriteFailed(RuntimeError):
    pass


def write(app, sheet, log, step, cell, prop, value, busy_timeout=15):
    """Set Range(cell).<prop> = value once Excel is ready; log it; raise on failure."""
    entry = {"step": step, "cell": cell, "property": prop,
             "formula": value if isinstance(value, str) else f"<{len(value)} rows of data>"}
    deadline = time.time() + busy_timeout
    attempts = 0
    while True:
        attempts += 1
        try:
            entry["ready_wait_s"] = wait_ready(app)
            setattr(sheet.range(cell).api, prop, value)
            entry["outcome"] = "ok"
            break
        except Exception as exc:
            if is_busy(exc) and time.time() < deadline:
                time.sleep(0.5)
                continue
            entry["outcome"] = f"error: {exc!r}"
            break
    entry["attempts"] = attempts
    log.append(entry)
    print("write " + json.dumps(entry, default=str), flush=True)
    if entry["outcome"] != "ok":
        raise WriteFailed(f"{step} write to {cell} failed: {entry['outcome']}")


def excel_processes():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='EXCEL.EXE'\" | "
           "Select-Object ProcessId, ParentProcessId | ConvertTo-Json")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout.strip()
    data = json.loads(out) if out else []
    return data if isinstance(data, list) else [data]


def python_processes():
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
           "Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine | ConvertTo-Json")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout.strip()
    data = json.loads(out) if out else []
    return data if isinstance(data, list) else [data]


def dialogs():
    """Visible dialog boxes (class #32770 or NUIDialog) of EXCEL.EXE, with their text."""
    user32 = ctypes.windll.user32
    pids = {p["ProcessId"] for p in excel_processes()}
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def text(hwnd):
        buf = ctypes.create_unicode_buffer(1024)
        user32.GetWindowTextW(hwnd, buf, 1024)
        return buf.value

    def cls(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    found = []

    def top(hwnd, _):
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd) and cls(hwnd) in ("#32770", "NUIDialog"):
            children = []
            user32.EnumChildWindows(hwnd, enum_proc(lambda h, _: children.append(text(h)) or True), 0)
            found.append({"class": cls(hwnd), "title": text(hwnd),
                          "text": [t for t in children if t]})
        return True

    user32.EnumWindows(enum_proc(top), 0)
    return found


def wait_excel_exit(timeout=90):
    deadline = time.time() + timeout
    while excel_processes():
        if time.time() > deadline:
            return False
        time.sleep(1)
    return True


def make_blank_book(path):
    app = xw.App(visible=False, add_book=False)   # automation instance: loads no add-ins
    try:
        book = app.books.add()
        book.save(str(path))
        book.close()
    finally:
        app.quit()
    wait_excel_exit()


def start_excel(mode, book_path):
    if excel_processes():
        raise SystemExit("Close Excel before running the test.")
    env = dict(os.environ)
    if mode == "without-home":
        env = {k: v for k, v in env.items() if k.upper() != "EXTRAPLUS_HOME"}
    return subprocess.Popen([excel_exe(), str(book_path)], env=env)


def attach(book_path, timeout=90):
    def find():
        for app in xw.apps:
            for book in app.books:
                if Path(book.fullname).resolve() == book_path.resolve():
                    return app, book
        raise RuntimeError("workbook not open yet")
    return retry(find, timeout=timeout, what="Excel to open the workbook")


def addin_state(app):
    def get():
        wb = app.api.Workbooks("extraplus.xlam")
        return {"fullname": wb.FullName,
                "env_source": app.api.Run("'extraplus.xlam'!ExtraplusEnvSource")}
    try:
        return retry(get, timeout=60, what="extraplus.xlam to load")
    except RuntimeError as exc:
        return {"error": str(exc)}


def read_cells(sheet, addresses, timeout):
    """Values once no cell is still waiting for the UDF server."""
    def ready():
        values = {a: sheet.range(a).value for a in addresses}
        if any(v is None or (isinstance(v, str) and v.startswith("#N/A")) for v in values.values()):
            raise RuntimeError(f"not ready: {values}")
        return values
    return retry(ready, timeout=timeout, what="the Extraplus functions")


def compare(values):
    x, y = zip(*DATA)
    rows, max_abs, all_equal = [], 0.0, True
    for n in MODELS:
        ref = reference(x, y, n)
        for kind, cell, expected in (("EXTRAP", f"D{1 + n}", ref[0]),
                                     ("UNCERTAINTY", f"D{4 + n}", ref[1]),
                                     ("DECAY_RATE", f"D{7 + n}", ref[2])):
            got = values[cell]
            ok = isinstance(got, float)
            diff = abs(got - expected) if ok else math.inf
            max_abs = max(max_abs, diff)
            all_equal &= ok and got == expected
            rows.append({"function": f"{kind}{n}", "cell": cell, "excel": got,
                         "direct": expected, "abs_diff": diff})
    return {"rows": rows, "max_abs_diff": max_abs, "bit_identical": all_equal}


def single_cell_step(app, book, results, mode):
    sheet = book.sheets[0]
    log = results["writes"]
    if mode == "new":
        data = [["Basis Size", "Energy"], *[[a, b] for a, b in DATA]]
        write(app, sheet, log, "data", f"A1:B{N + 1}", "Value", data)
        for cell, formula in CORE.items():
            write(app, sheet, log, "single-cell", cell, "Formula", formula)
    else:
        results["link_sources"] = list(book.api.LinkSources(1) or [])  # xlExcelLinks
        wait_ready(app)
        app.api.CalculateFull()
    results["formulas_as_stored"] = {c: sheet.range(c).api.Formula for c in CORE}
    started = time.time()
    values = read_cells(sheet, list(CORE), timeout=180)
    results["seconds_to_values"] = round(time.time() - started, 1)
    results["values"] = values
    results["version_cell"] = values["D11"]
    results["error_cell"] = values["D12"]
    results["error_cell_ok"] = isinstance(values["D12"], str) and values["D12"].startswith("Error: ")
    results["comparison"] = compare(values)


def spill_step(app, book, results, mode):
    """EXTRAP1_V / EXTRAP1_H: legacy array output (dynamic-array spill is not
    tested via COM, see the module docstring). Recorded as observed."""
    sheet = book.sheets[0]
    out = results["spill"] = {"note": "legacy array output; dynamic-array spill not tested via COM"}
    expected = [results["values"]["D2"], results["values"]["D5"], results["values"]["D8"]]
    for anchor, (formula, cells) in SPILL.items():
        r = out[anchor] = {"formula": formula, "cells": cells}
        try:
            if mode == "new":
                write(app, sheet, results["writes"], "spill", anchor, "Formula", formula)
            try:
                r["values"] = list(read_cells(sheet, cells, timeout=30).values())
            except RuntimeError as exc:
                r["values"] = [sheet.range(c).value for c in cells]
                r["read_error"] = str(exc)
            r["matches_single_cell"] = r["values"] == expected
            r["formula_stored"] = {c: sheet.range(c).api.Formula for c in cells}
            r["has_array"] = {c: bool(sheet.range(c).api.HasArray) for c in cells}
            if all(r["has_array"].values()):
                r["array_range"] = sheet.range(anchor).api.CurrentArray.Address
        except Exception:
            r["exception"] = traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--launch", choices=["inherit", "without-home"], default="inherit")
    parser.add_argument("--mode", choices=["new", "reopen", "startup"], default="new")
    args = parser.parse_args()
    args.book = args.book.resolve()

    results = {"launch": args.launch, "mode": args.mode, "book": str(args.book), "inputs": DATA,
               "harness_pid": os.getpid(), "writes": []}
    app = None
    try:
        if args.mode == "new" and not args.book.exists():
            make_blank_book(args.book)
        proc = start_excel(args.launch, args.book)
        results["launched_pid"] = proc.pid
        if args.mode == "startup":
            time.sleep(20)                       # let Excel load its add-ins
            results["dialogs"] = dialogs()
            try:
                app, book = attach(args.book, timeout=30)
                try:                             # add-ins are not listed when iterating Workbooks
                    results["extraplus_loaded"] = app.api.Workbooks("extraplus.xlam").FullName
                except Exception:
                    results["extraplus_loaded"] = False
            except RuntimeError as exc:          # a modal dialog blocks COM
                results["attach_error"] = str(exc)
                subprocess.run(["taskkill.exe", "/PID", str(proc.pid), "/F"], capture_output=True)
        else:
            app, book = attach(args.book)
            app.display_alerts = False
            results["excel_version"] = app.version
            results["excel_build"] = app.api.Build
            results["excel_processes"] = excel_processes()
            results["addin"] = addin_state(app)
            results["dialogs"] = dialogs()
            single_cell_step(app, book, results, args.mode)
            spill_step(app, book, results, args.mode)
            results["python_processes"] = python_processes()
            if args.mode == "new":
                wait_ready(app)
                book.save()
    except Exception:
        results["exception"] = traceback.format_exc()
    finally:
        if app is not None:
            try:
                app.quit()
            except Exception:
                pass
        results["excel_exited"] = wait_excel_exit()
        args.results.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    c = results.get("comparison", {})
    print(json.dumps({k: results.get(k) for k in (
        "addin", "dialogs", "extraplus_loaded", "attach_error", "version_cell", "error_cell",
        "error_cell_ok", "seconds_to_values", "spill", "link_sources", "excel_exited",
        "exception")}, indent=1, default=str))
    print(f"bit_identical={c.get('bit_identical')} max_abs_diff={c.get('max_abs_diff')}")


if __name__ == "__main__":
    main()
