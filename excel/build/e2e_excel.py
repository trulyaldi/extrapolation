"""
End-to-end test of an installed Extraplus in a real Excel (Windows only).

Run with the INSTALLED environment's Python, so the reference values come from
the same packages the add-in uses:

    %EXTRAPLUS_HOME%\\venv\\Scripts\\python.exe e2e_excel.py --book B.xlsx --results R.json [options]

Excel is started the way a user starts it, by opening --book (a COM automation
instance would not load add-ins), and then attached to:

    --launch explorer   through the Windows shell (Explorer), which has the
                        current user environment including EXTRAPLUS_HOME
    --launch direct     as a child of this process without EXTRAPLUS_HOME, to
                        exercise the add-in's registry fallback

--mode new (default) creates --book if needed, writes the test data and
formulas, compares every cell with extraplus called directly, and saves the
book. --mode reopen only recalculates an existing book and compares.
"""

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import traceback
import winreg
from pathlib import Path

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
SPILL = {"F2": f"=EXTRAP1_V({X},{Y})"}   # best effort: F2:F4


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


def shell_pid():
    """PID of the Explorer process that owns the taskbar (the user's shell)."""
    hwnd = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


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


def launch(mode, book_path):
    if excel_processes():
        raise SystemExit("Close Excel before running the test.")
    if mode == "explorer":
        subprocess.Popen(["explorer.exe", str(book_path)])
    else:
        env = {k: v for k, v in os.environ.items() if k.upper() != "EXTRAPLUS_HOME"}
        subprocess.Popen([excel_exe(), str(book_path)], env=env)

    def attach():
        for app in xw.apps:
            for book in app.books:
                if Path(book.fullname).resolve() == book_path.resolve():
                    return app, book
        raise RuntimeError("workbook not open yet")
    return retry(attach, what="Excel to open the workbook")


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


def test_book(app, book, results, mode):
    sheet = book.sheets[0]
    if mode == "new":
        sheet.range("A1").value = [["Basis Size", "Energy"], *[[a, b] for a, b in DATA]]
        for cell, formula in {**CORE, **SPILL}.items():
            sheet.range(cell).api.Formula2 = formula
    else:
        results["link_sources"] = list(book.api.LinkSources(1) or [])  # xlExcelLinks
        app.api.CalculateFull()
    results["formulas_as_stored"] = {c: sheet.range(c).api.Formula2 for c in {**CORE, **SPILL}}
    started = time.time()
    values = read_cells(sheet, [*CORE, "F2"], timeout=180)
    results["seconds_to_values"] = round(time.time() - started, 1)
    results["values"] = values
    results["version_cell"] = values["D11"]
    results["error_cell"] = values["D12"]
    results["comparison"] = compare(values)
    try:
        spill = list(read_cells(sheet, ["F2", "F3", "F4"], timeout=20).values())
        results["spill_F2_F4"] = spill
        results["spill_matches_D"] = spill == [values["D2"], values["D5"], values["D8"]]
    except RuntimeError as exc:
        results["spill_F2_F4"] = f"did not spill: {exc}"
    if mode == "new":
        book.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--launch", choices=["explorer", "direct"], default="explorer")
    parser.add_argument("--mode", choices=["new", "reopen"], default="new")
    args = parser.parse_args()
    args.book = args.book.resolve()

    results = {"launch": args.launch, "mode": args.mode, "book": str(args.book), "inputs": DATA}
    app = None
    try:
        if args.mode == "new" and not args.book.exists():
            make_blank_book(args.book)
        app, book = launch(args.launch, args.book)
        app.display_alerts = False
        results["excel_version"] = app.version
        results["excel_parent_is_shell"] = [p["ParentProcessId"] == shell_pid() for p in excel_processes()]
        results["addin"] = addin_state(app)
        test_book(app, book, results, args.mode)
        results["python_processes"] = python_processes()
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
        "addin", "excel_parent_is_shell", "version_cell", "error_cell", "seconds_to_values",
        "spill_F2_F4", "link_sources", "excel_exited", "exception")}, indent=1, default=str))
    print(f"bit_identical={c.get('bit_identical')} max_abs_diff={c.get('max_abs_diff')}")


if __name__ == "__main__":
    main()
