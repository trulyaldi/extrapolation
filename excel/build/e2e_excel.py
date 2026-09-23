"""
End-to-end test of an installed Extraplus in a real Excel (Windows only).

Run with the INSTALLED environment's Python so the reference values come from the
same packages the add-in uses:

    %EXTRAPLUS_HOME%\\venv\\Scripts\\python.exe e2e_excel.py --results r.json [options]

Excel is started the way a user starts it (not as a COM automation instance,
which would not load add-ins) and then attached to:

    --launch explorer   through Explorer, which has the current user environment
    --launch direct     directly from this process, without EXTRAPLUS_HOME, to
                        exercise the add-in's registry fallback

Default test: new workbook, data in A1:B13, formulas in column D and F, then
compare every cell with extraplus called directly. --save stores the workbook as
.xlsx; --reopen PATH instead opens an existing workbook and recalculates it.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import winreg
from pathlib import Path

import xlwings as xw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_reference import DATA, MODELS, reference  # noqa: E402

N = len(DATA)
X = f"A2:A{N + 1}"
Y = f"B2:B{N + 1}"
FORMULAS = {
    "D2": f"=EXTRAP1({X},{Y})", "D3": f"=EXTRAP2({X},{Y})", "D4": f"=EXTRAP3({X},{Y})",
    "D5": f"=UNCERTAINTY1({X},{Y})", "D6": f"=UNCERTAINTY2({X},{Y})", "D7": f"=UNCERTAINTY3({X},{Y})",
    "D8": f"=DECAY_RATE1({X},{Y})", "D9": f"=DECAY_RATE2({X},{Y})", "D10": f"=DECAY_RATE3({X},{Y})",
    "D11": "=EXTRAPLUS_VERSION()",
    "D12": "=EXTRAP1(A2:A3,B2:B3)",
    "F2": f"=EXTRAP1_V({X},{Y})",
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
        except Exception as exc:  # Excel is busy while starting / calculating
            if time.time() > deadline:
                raise RuntimeError(f"timed out waiting for {what}: {exc!r}") from exc
            time.sleep(1)


def launch(mode):
    if xw.apps.count:
        raise SystemExit("Close Excel before running the test.")
    exe = excel_exe()
    if mode == "explorer":
        subprocess.Popen(["explorer.exe", exe])
    else:
        env = {k: v for k, v in os.environ.items() if k.upper() != "EXTRAPLUS_HOME"}
        subprocess.Popen([exe], env=env)

    def attach():
        app = xw.apps.active
        app.api.Version  # noqa: B018 - raises until Excel accepts COM calls
        return app
    return retry(attach, what="Excel to start")


def addin_state(app):
    def get():
        wb = app.api.Workbooks("extraplus.xlam")
        return {"fullname": wb.FullName,
                "env_source": app.api.Run("'extraplus.xlam'!ExtraplusEnvSource")}
    try:
        return retry(get, timeout=60, what="extraplus.xlam to load")
    except RuntimeError as exc:
        return {"error": str(exc)}


def python_processes():
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
           "Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine | ConvertTo-Json")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True).stdout.strip()
    data = json.loads(out) if out else []
    return data if isinstance(data, list) else [data]


def read_cells(sheet, addresses, timeout=180):
    """Values once no cell is still waiting for the UDF server."""
    def ready():
        values = {a: sheet.range(a).value for a in addresses}
        if any(v is None or (isinstance(v, str) and v.startswith("#N/A")) for v in values.values()):
            raise RuntimeError(f"not ready: {values}")
        return values
    return retry(ready, timeout=timeout, what="the Extraplus functions")


def compare(values):
    x, y = zip(*DATA)
    rows, max_abs, max_rel, all_equal = [], 0.0, 0.0, True
    for n in MODELS:
        ref = reference(x, y, n)
        for kind, cell, expected in (("EXTRAP", f"D{1 + n}", ref[0]),
                                     ("UNCERTAINTY", f"D{4 + n}", ref[1]),
                                     ("DECAY_RATE", f"D{7 + n}", ref[2])):
            got = values[cell]
            ok = isinstance(got, float)
            diff = abs(got - expected) if ok else math.inf
            rel = diff / abs(expected) if ok and expected else diff
            max_abs, max_rel = max(max_abs, diff), max(max_rel, rel)
            all_equal &= ok and got == expected
            rows.append({"function": f"{kind}{n}", "cell": cell, "excel": got,
                         "direct": expected, "abs_diff": diff})
    spill = [values["F2"], values["F3"], values["F4"]]
    return {"rows": rows, "max_abs_diff": max_abs, "max_rel_diff": max_rel,
            "bit_identical": all_equal, "spill_F2_F4": spill,
            "spill_matches_D": spill == [values["D2"], values["D5"], values["D8"]]}


def run_new(app, results, save):
    book = app.books.add()
    sheet = book.sheets[0]
    sheet.range("A1").value = [["Basis Size", "Energy"], *[[a, b] for a, b in DATA]]
    for cell, formula in FORMULAS.items():
        sheet.range(cell).formula = formula
    app.calculate()
    cells = [*FORMULAS, "F3", "F4"]
    started = time.time()
    values = read_cells(sheet, cells)
    results["seconds_to_values"] = round(time.time() - started, 1)
    results["values"] = values
    results["formulas_as_stored"] = {c: sheet.range(c).formula for c in FORMULAS}
    results["comparison"] = compare(values)
    results["version_cell"] = values["D11"]
    results["error_cell"] = values["D12"]
    if save:
        book.save(str(save))
        results["saved"] = str(save)
    book.close()


def run_reopen(app, results, path):
    book = retry(lambda: app.books.open(str(path)), what="the workbook to open")
    sheet = book.sheets[0]
    results["formulas_as_stored"] = {c: sheet.range(c).formula for c in FORMULAS}
    results["link_sources"] = list(book.api.LinkSources(1) or [])  # xlExcelLinks
    app.api.CalculateFull()
    results["values"] = read_cells(sheet, [*FORMULAS, "F3", "F4"])
    results["comparison"] = compare(results["values"])
    book.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--launch", choices=["explorer", "direct"], default="explorer")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reopen", type=Path)
    args = parser.parse_args()

    results = {"launch": args.launch, "inputs": DATA}
    app = launch(args.launch)
    try:
        app.display_alerts = False
        results["excel_version"] = app.version
        results["addin"] = addin_state(app)
        if args.reopen:
            run_reopen(app, results, args.reopen)
        else:
            run_new(app, results, args.save)
        results["python_processes"] = python_processes()
    finally:
        try:
            app.quit()
        except Exception:
            pass
    args.results.write_text(json.dumps(results, indent=2, default=str))
    c = results.get("comparison", {})
    print(json.dumps({k: results.get(k) for k in ("addin", "version_cell", "error_cell", "seconds_to_values")}, default=str))
    print(f"bit_identical={c.get('bit_identical')} max_abs_diff={c.get('max_abs_diff')} "
          f"spill_matches_D={c.get('spill_matches_D')}")


if __name__ == "__main__":
    main()
