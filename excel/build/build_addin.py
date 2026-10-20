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
  4. add the EXTRAPLUS_HOME fallback (ENV_MODULE), the relink-on-open and
     pre-save guard (EVENTS_CLASS, EVENTS_MODULE) and the ribbon command
     (COMMANDS_MODULE), and import the UDF wrappers generated from
     extrapolation.py;
  5. save, add the ribbon (CUSTOM_UI) to the saved file, reopen it to check
     the result, and write extraplus.xlam.buildinfo.json for make_release.sh.
"""

import argparse
import hashlib
import io
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

# Workbooks remember the add-in by its full path (xl/externalLinks/), and the
# path is per user (%LOCALAPPDATA%\Extraplus). When a workbook opens whose
# links point to an extraplus.xlam somewhere else, repoint them to this one;
# otherwise its formulas become #NAME? on the next calculation.
# ExtraplusLinkLog() reports what was done (used by the tests).
EVENTS_CLASS = r'''Option Explicit

Public WithEvents App As Application

Private Sub App_WorkbookOpen(ByVal Wb As Workbook)
    RelinkWorkbook Wb
End Sub

Private Sub App_WorkbookBeforeSave(ByVal Wb As Workbook, ByVal SaveAsUI As Boolean, Cancel As Boolean)
    If Not ConfirmSave(Wb) Then Cancel = True
End Sub
'''

EVENTS_MODULE = r'''Option Explicit
Option Private Module

' Warn before saving when at least this many cells with Extraplus formulas
' show #NAME?, #REF! or #VALUE! (a missing add-in or a failed calculation).
Private Const GUARD_MIN_ERRORS As Long = 1

Private mEvents As ExtraplusEvents
Private mLog As String

Public Sub StartExtraplusEvents()
    Dim wb As Workbook
    Set mEvents = New ExtraplusEvents
    Set mEvents.App = Application
    For Each wb In Application.Workbooks   ' opened before the add-in was loaded
        RelinkWorkbook wb
    Next wb
End Sub

Private Function FileNameOf(ByVal path As String) As String
    Dim i As Long
    i = InStrRev(Replace(path, "/", "\"), "\")
    FileNameOf = Mid$(path, i + 1)
End Function

Public Sub RelinkWorkbook(ByVal wb As Workbook)
    Dim links As Variant, i As Long, link As String, target As String, changed As Long, failed As Long
    On Error GoTo finish
    If wb Is ThisWorkbook Then Exit Sub
    If wb.IsAddin Then Exit Sub
    links = wb.LinkSources(xlExcelLinks)
    If IsEmpty(links) Then GoTo finish
    target = ThisWorkbook.FullName
    For i = LBound(links) To UBound(links)
        link = CStr(links(i))
        If StrComp(FileNameOf(link), ThisWorkbook.Name, vbTextCompare) = 0 And _
           StrComp(link, target, vbTextCompare) <> 0 Then
            On Error Resume Next
            wb.ChangeLink link, target, xlLinkTypeExcelLinks
            If Err.Number = 0 Then changed = changed + 1 Else failed = failed + 1
            Err.Clear
            On Error GoTo finish
        End If
    Next i
finish:
    mLog = mLog & wb.Name & ": relinked " & changed & ", failed " & failed & vbLf
End Sub

Public Function ExtraplusLinkLog() As String
    ExtraplusLinkLog = mLog
End Function

Private Function AutoSaveIsOn(ByVal wb As Workbook) As Boolean
    On Error Resume Next   ' the property does not exist in every Excel
    AutoSaveIsOn = wb.AutoSaveOn
End Function

Private Function IsExtraplusFormula(ByVal formula As String) As Boolean
    formula = UCase$(formula)
    IsExtraplusFormula = InStr(formula, "EXTRAP") > 0 Or InStr(formula, "UNCERTAINTY") > 0 Or _
                         InStr(formula, "DECAY_RATE") > 0
End Function

Public Function CountBrokenCells(ByVal wb As Workbook, ByRef example As String) As Long
    Dim ws As Worksheet, found As Range, cell As Range, v As Variant
    For Each ws In wb.Worksheets
        Set found = Nothing
        On Error Resume Next   ' no error cells, or a protected sheet
        Set found = ws.UsedRange.SpecialCells(xlCellTypeFormulas, xlErrors)
        On Error GoTo 0
        If Not found Is Nothing Then
            For Each cell In found.Cells
                v = cell.Value
                If IsError(v) Then
                    If v = CVErr(xlErrName) Or v = CVErr(xlErrRef) Or v = CVErr(xlErrValue) Then
                        If IsExtraplusFormula(cell.Formula) Then
                            CountBrokenCells = CountBrokenCells + 1
                            If example = "" Then example = "'" & ws.Name & "'!" & cell.Address(False, False)
                        End If
                    End If
                End If
            Next cell
        End If
    Next ws
End Function

' False only when the user, warned, chose not to save. Any problem here lets
' the save go ahead.
Public Function ConfirmSave(ByVal wb As Workbook) As Boolean
    Dim n As Long, example As String
    ConfirmSave = True
    On Error GoTo finish
    If wb Is ThisWorkbook Or wb.IsAddin Then Exit Function
    If Not Application.DisplayAlerts Or AutoSaveIsOn(wb) Then Exit Function
    n = CountBrokenCells(wb, example)
    If n < GUARD_MIN_ERRORS Then Exit Function
    ConfirmSave = (MsgBox(n & " cells with Extraplus formulas show #NAME?, #REF! or #VALUE! (for example " & _
        example & ")." & vbLf & vbLf & _
        "This usually means that the Extraplus add-in was not found or the calculation failed. " & _
        "Saving now replaces the numbers stored in the file with these errors." & vbLf & vbLf & _
        "Save anyway?", vbExclamation + vbYesNo + vbDefaultButton2, "Extraplus") = vbYes)
finish:
End Function
'''

# The ribbon button (CUSTOM_UI) calls this.
COMMANDS_MODULE = r'''Option Explicit

Public Sub ExtraplusWriteValues(Optional control As IRibbonControl)
    Dim scope As String
    If ActiveWorkbook Is Nothing Then Exit Sub
    If TypeName(Selection) = "Range" Then
        If Selection.CountLarge > 1 Then scope = "the selected cells"
    End If
    If scope = "" Then scope = "the sheet '" & ActiveSheet.Name & "'"
    If ActiveWorkbook.Path = "" Then
        MsgBox "Save the workbook first: a backup copy is saved next to it before any formula is replaced.", _
               vbInformation, "Extraplus"
        Exit Sub
    End If
    If MsgBox("Replace the Extraplus formulas in " & scope & " with the numbers they compute?" & vbLf & vbLf & _
              "The numbers are computed again from the formulas' input cells. Before anything is replaced, " & _
              "a copy of the workbook is saved next to it, and each replaced formula is listed on the " & _
              "sheet 'Extraplus values'. Formulas that give an error are kept.", _
              vbQuestion + vbOKCancel + vbDefaultButton2, "Extraplus") <> vbOK Then Exit Sub
    RunPython "import extrapolation; extrapolation.write_values()"
End Sub
'''

WORKBOOK_OPEN = '''Private Sub Workbook_Open()
    EnsureExtraplusHome
    StartExtraplusEvents
End Sub
'''

# Office 2007 RibbonX (works in every later Excel), added to the saved file.
CUSTOM_UI = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<customUI xmlns="http://schemas.microsoft.com/office/2006/01/customui">
  <ribbon>
    <tabs>
      <tab id="extraplusTab" label="Extraplus">
        <group id="extraplusResults" label="Results">
          <button id="extraplusWriteValues" label="Write values" size="large" imageMso="PasteValues"
                  onAction="ExtraplusWriteValues"
                  screentip="Replace Extraplus formulas by their values"
                  supertip="Computes the Extraplus formulas in the selection (or on the sheet) again and replaces them with the numbers, so the sheet no longer recalculates them. A backup copy of the workbook is saved first."/>
        </group>
      </tab>
    </tabs>
  </ribbon>
</customUI>
'''
CUSTOM_UI_PART = "customUI/customUI.xml"
CUSTOM_UI_REL = ('<Relationship Id="rIdExtraplusUI" '
                 'Type="http://schemas.microsoft.com/office/2006/relationships/ui/extensibility" '
                 f'Target="{CUSTOM_UI_PART}"/>')

VBEXT_CT_STDMODULE = 1
VBEXT_CT_CLASSMODULE = 2


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
        for kind, name, code in ((VBEXT_CT_STDMODULE, "extraplus_env", ENV_MODULE),
                                 (VBEXT_CT_CLASSMODULE, "ExtraplusEvents", EVENTS_CLASS),
                                 (VBEXT_CT_STDMODULE, "extraplus_events", EVENTS_MODULE),
                                 (VBEXT_CT_STDMODULE, "extraplus_commands", COMMANDS_MODULE)):
            component = vbproject.VBComponents.Add(kind)
            component.Name = name
            component.CodeModule.AddFromString(code)
        components["ThisWorkbook"].CodeModule.AddFromString(WORKBOOK_OPEN)

        sheet = book.sheets["_myaddin.conf"]
        sheet.name = f"{PROJECT}.conf"
        sheet.clear()
        sheet.range("A:B").number_format = "@"   # keep "False" etc. as text
        sheet.range("A1").value = [list(row) for row in CONFIG]

        # Application.MacroOptions (function descriptions, run by import_udfs)
        # refuses to edit a hidden workbook, so show the add-in while importing.
        book.api.IsAddin = False
        udfs.import_udfs(UDF_MODULE, book.api)
        book.api.IsAddin = True

        props = book.api.BuiltinDocumentProperties
        props("Title").Value = "Extraplus"
        props("Comments").Value = (
            f"Extraplus {version('extraplus')}: extrapolation functions EXTRAP1-3, "
            "UNCERTAINTY1-3, DECAY_RATE1-3, EXTRAP_ALL_V and more (category Extrapolation)."
        )
        book.save()
        book.close()
    finally:
        app.quit()
    add_custom_ui(xlam)
    return xlam


def add_custom_ui(xlam):
    """Add the ribbon (CUSTOM_UI) to the saved add-in; Excel keeps it on later saves."""
    import zipfile
    data = Path(xlam).read_bytes()
    temp = Path(xlam).with_suffix(".tmp")
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as dst:
        if CUSTOM_UI_PART in src.namelist():
            raise RuntimeError("the add-in already has a ribbon part")
        for item in src.infolist():
            content = src.read(item.filename)
            if item.filename == "_rels/.rels":
                text = content.decode("utf-8")
                if "</Relationships>" not in text:
                    raise RuntimeError("unexpected _rels/.rels")
                content = text.replace("</Relationships>", CUSTOM_UI_REL + "</Relationships>").encode("utf-8")
            dst.writestr(item, content)
        dst.writestr(CUSTOM_UI_PART, CUSTOM_UI.encode("utf-8"))
    temp.replace(xlam)


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
        this_workbook = components["ThisWorkbook"].CodeModule
        workbook_code = this_workbook.Lines(1, this_workbook.CountOfLines) if this_workbook.CountOfLines else ""
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
            "has_workbook_open": "EnsureExtraplusHome" in workbook_code,
            "starts_events": "StartExtraplusEvents" in workbook_code,
            "sheets": [s.name for s in book.sheets],
        }
        book.close()
    finally:
        app.quit()
    import zipfile
    with zipfile.ZipFile(xlam) as z:
        info["custom_ui"] = CUSTOM_UI_PART in z.namelist() and CUSTOM_UI_REL in z.read("_rels/.rels").decode()
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
    if not {"EXTRAPLUS_VERSION", "EXTRAP1", "EXTRAP_ALL_V", "EXTRAPLUS_CACHE_STATS"} <= set(info["functions"]):
        problems.append(f"functions missing: {info['functions']}")
    if not info["is_addin"]:
        problems.append("the workbook is not saved as an add-in (IsAddin is False)")
    if "extraplus_env" not in info["components"] or not info["has_workbook_open"]:
        problems.append("the EXTRAPLUS_HOME fallback (extraplus_env / Workbook_Open) is missing")
    if not {"ExtraplusEvents", "extraplus_events", "extraplus_commands"} <= set(info["components"]) \
            or not info["starts_events"]:
        problems.append("the relink / save guard / ribbon VBA is missing")
    if not info["custom_ui"]:
        problems.append("the ribbon part (customUI) is missing")
    if "Module1" in info["components"]:
        problems.append("the template's Module1 was not removed")
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
