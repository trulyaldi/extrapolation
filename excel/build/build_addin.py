"""
Build extraplus.xlam: the Excel add-in that makes the Extraplus functions
available in every workbook.

Runs on Windows in the build environment set up by build_addin.ps1 and drives
a hidden Excel instance (automation instances load no add-ins or XLSTART
files). Excel must allow "Trust access to the VBA project object model"
(File > Options > Trust Center > Trust Center Settings > Macro Settings).

Steps:
  1. copy xlwings' custom add-in template (quickstart_addin.xlam, which embeds
     the xlwings VBA and needs no xlwings add-in or VBA reference);
  2. rename it to the "extraplus" project: PROJECT_NAME, VBA project name and
     the "extraplus.conf" config sheet (both must change together);
  3. fill the config sheet with every key xlwings reads when it runs a UDF, so
     no lookup falls through to a workbook's folder or a user config file;
  4. add the EXTRAPLUS_HOME fallback (see ENV_MODULE) and import the UDF
     wrappers generated from extrapolation.py;
  5. save and write extraplus.xlam.buildinfo.json for make_release.sh.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

import xlwings as xw
from xlwings import udfs

PROJECT = "extraplus"
UDF_MODULE = "extrapolation"

# Keys are matched case-insensitively by xlwings. Values are stored as text.
CONFIG = [
    ("INTERPRETER_WIN", r"%EXTRAPLUS_HOME%\venv\Scripts\pythonw.exe"),
    ("INTERPRETER", ""),
    ("PYTHONPATH", r"%EXTRAPLUS_HOME%\udf"),
    ("UDF MODULES", UDF_MODULE),
    # Never put the workbook's folder on sys.path: an old extrapolation.py
    # next to a workbook would shadow the installed module.
    ("ADD_WORKBOOK_TO_PYTHONPATH", "False"),
    ("USE UDF SERVER", "True"),
    ("DEBUG UDFS", "False"),
    ("SHOW CONSOLE", "False"),
    ("CONDA PATH", ""),
    ("CONDA ENV", ""),
    ("LICENSE_KEY", ""),
    ("ONEDRIVE_CONSUMER_WIN", ""),
    ("ONEDRIVE_COMMERCIAL_WIN", ""),
    ("SHAREPOINT_WIN", ""),
]

# The config above refers to %EXTRAPLUS_HOME%, which install.ps1 stores in
# HKCU\Environment. Excel started by a program that was already running before
# the install (and missed the change notification) does not have it, so copy it
# from the registry into Excel's own environment when the add-in loads.
# ExtraplusEnvSource() reports where the value came from (used by the tests).
ENV_MODULE = r'''Option Explicit
Option Private Module

Private mSource As String

Public Sub EnsureExtraplusHome()
    Dim shell As Object, value As String
    On Error Resume Next
    Set shell = CreateObject("WScript.Shell")
    If Len(shell.Environment("Process").Item("EXTRAPLUS_HOME")) > 0 Then
        mSource = "inherited"
    Else
        value = shell.RegRead("HKCU\Environment\EXTRAPLUS_HOME")
        If Len(value) > 0 Then
            shell.Environment("Process").Item("EXTRAPLUS_HOME") = value
            mSource = "registry"
        Else
            mSource = "missing"
        End If
    End If
End Sub

Public Function ExtraplusEnvSource() As String
    ExtraplusEnvSource = mSource & "|" & CreateObject("WScript.Shell").ExpandEnvironmentStrings("%EXTRAPLUS_HOME%")
End Function
'''

WORKBOOK_OPEN = '''Private Sub Workbook_Open()
    EnsureExtraplusHome
End Sub
'''

VBEXT_CT_STDMODULE = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_line(code_module, old, new):
    for line in range(1, code_module.CountOfLines + 1):
        if code_module.Lines(line, 1).strip() == old:
            code_module.ReplaceLine(line, new)
            return
    raise RuntimeError(f"line not found in VBA module: {old}")


def vba_components(book):
    return {c.Name: c for c in book.api.VBProject.VBComponents}


def build(out_dir, udf_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    xlam = out_dir / f"{PROJECT}.xlam"
    template = Path(xw.__file__).parent / "quickstart_addin.xlam"
    shutil.copyfile(template, xlam)

    sys.path.insert(0, str(udf_dir))
    app = xw.App(visible=False, add_book=False)
    try:
        app.display_alerts = False
        app.screen_updating = False
        book = app.books.open(str(xlam))
        try:
            vbproject = book.api.VBProject
        except Exception as exc:
            raise SystemExit(
                "Excel refused access to the VBA project. Enable File > Options > Trust Center > "
                "Trust Center Settings > Macro Settings > 'Trust access to the VBA project object model'."
            ) from exc

        vbproject.Name = PROJECT
        components = vba_components(book)
        replace_line(components["xlwings"].CodeModule,
                     'Public Const PROJECT_NAME As String = "myaddin"',
                     f'Public Const PROJECT_NAME As String = "{PROJECT}"')
        vbproject.VBComponents.Remove(components["Module1"])  # template's sample macro
        env = vbproject.VBComponents.Add(VBEXT_CT_STDMODULE)
        env.Name = "extraplus_env"
        env.CodeModule.AddFromString(ENV_MODULE)
        components["ThisWorkbook"].CodeModule.AddFromString(WORKBOOK_OPEN)

        sheet = book.sheets["_myaddin.conf"]
        sheet.name = f"{PROJECT}.conf"
        sheet.clear()
        sheet.range("A:B").number_format = "@"   # keep "False" etc. as text
        sheet.range("A1").value = [list(row) for row in CONFIG]

        udfs.import_udfs(UDF_MODULE, book.api)

        props = book.api.BuiltinDocumentProperties
        props("Title").Value = "Extraplus"
        props("Comments").Value = (
            f"Extraplus {version('extraplus')}: extrapolation functions EXTRAP1-3, "
            "UNCERTAINTY1-3, DECAY_RATE1-3 and more (category Extrapolation)."
        )
        book.save()
        book.close()
    finally:
        app.quit()
    return xlam


def inspect_xlam(xlam, out_dir):
    """Reopen the add-in and check what was built."""
    app = xw.App(visible=False, add_book=False)
    try:
        app.display_alerts = False
        book = app.books.open(str(xlam))
        components = vba_components(book)
        udf_code = components["xlwings_udfs"].CodeModule
        code = udf_code.Lines(1, udf_code.CountOfLines) if udf_code.CountOfLines else ""
        (out_dir / "xlwings_udfs.bas").write_text(code, encoding="utf-8")
        xlwings_code = components["xlwings"].CodeModule
        sheet = book.sheets[f"{PROJECT}.conf"]
        config = sheet.range("A1").options(ndim=2).expand("down").resize(None, 2).value
        info = {
            "components": sorted(components),
            "functions": sorted(
                line.split("(")[0].split()[1] for line in code.splitlines() if line.startswith("Function ")
            ),
            "volatile_count": code.count("Application.Volatile"),
            "project_name_line": next(
                xlwings_code.Lines(i, 1).strip()
                for i in range(1, xlwings_code.CountOfLines + 1)
                if "PROJECT_NAME As String" in xlwings_code.Lines(i, 1)
            ),
            "config": [[k, v if v is not None else ""] for k, v in config],
            "is_addin": bool(book.api.IsAddin),
            "sheets": [s.name for s in book.sheets],
        }
        book.close()
    finally:
        app.quit()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--udf-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="folder that contains extrapolation.py")
    args = parser.parse_args()

    xlam = build(args.out, args.udf_dir)
    time.sleep(2)  # let the first Excel instance exit
    info = inspect_xlam(xlam, args.out)

    expected_config = [[k, v] for k, v in CONFIG]
    problems = []
    if info["volatile_count"]:
        problems.append("a generated UDF is volatile")
    if info["config"] != expected_config:
        problems.append(f"config sheet mismatch: {info['config']}")
    if info["project_name_line"] != f'Public Const PROJECT_NAME As String = "{PROJECT}"':
        problems.append(f"PROJECT_NAME not set: {info['project_name_line']}")
    if "EXTRAPLUS_VERSION" not in info["functions"] or "EXTRAP1" not in info["functions"]:
        problems.append(f"functions missing: {info['functions']}")
    if problems:
        raise SystemExit("Build check failed: " + "; ".join(problems))

    buildinfo = {
        "xlwings": version("xlwings"),
        "extraplus": version("extraplus"),
        "extrapolation_py_sha256": sha256(args.udf_dir / f"{UDF_MODULE}.py"),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **info,
    }
    (args.out / f"{PROJECT}.xlam.buildinfo.json").write_text(json.dumps(buildinfo, indent=2) + "\n")
    print(f"Built {xlam} with {len(info['functions'])} functions: {', '.join(info['functions'])}")


if __name__ == "__main__":
    main()
