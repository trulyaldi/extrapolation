# Extraplus for Excel

Extraplus adds extrapolation functions to Microsoft Excel on Windows, for example

```
=EXTRAP1(A2:A20, B2:B20)
```

They work in every workbook, including plain `.xlsx` files. You do not need Python,
admin rights, or any Excel settings.

**Requirements:** Windows 10 or 11, desktop Microsoft Excel (Microsoft 365, 2021,
2019 or 2016; 32-bit Office has not been tested), and an internet connection while
installing. Tested with Excel 2021 (16.0.14332, 64-bit) on Windows 10 21H2.

## Install

1. Close Excel.
2. Unzip `extraplus-excel-vX.Y.Z.zip` (right-click > Extract All).
3. Double-click `install.bat` in the unzipped folder. If Windows shows
   "Windows protected your PC", click **More info > Run anyway**.
4. Wait for `SUCCESS` (a few minutes the first time), then press Enter.

Open Excel and type a formula. The first calculation after starting Excel takes a
few seconds. You can delete the unzipped folder afterwards.

## Functions

`x` is the range of basis sizes, `y` the range of values (same number of cells;
blank and text cells are skipped). The digit selects the model:
**1** exponential, **2** square-root exponential, **3** power law.

| Function | Returns |
|---|---|
| `EXTRAP1(x, y, [b_fixed], [n_fit])` (also 2, 3) | extrapolated value C at infinite basis |
| `UNCERTAINTY1(x, y, [b_fixed], [n_fit], ...)` (also 2, 3) | uncertainty of C, in the units of y |
| `DECAY_RATE1(x, y, [b_fixed], [n_fit])` (also 2, 3) | fitted (or fixed) decay parameter B |
| `EXTRAP1_V(x, y, ...)` / `EXTRAP1_H(x, y, ...)` (also 2, 3) | C, uncertainty and B in 3 cells, down / across |
| `EXTRAPLUS_VERSION()` | installed Extraplus version |

`b_fixed` fixes B instead of fitting it; `n_fit` uses only the first `n_fit` points.
Invalid input shows a message such as `Error: At least three paired numeric x/y
observations are required.` The functions are listed under the **Extrapolation**
category in Insert Function.

## Update

Download the new zip, close Excel, and run its `install.bat`. Your workbooks are not
affected.

## Uninstall

Close Excel, then **Settings > Apps > Extraplus for Excel > Uninstall**. (Or run
`uninstall.bat` from the unzipped folder, or from the install folder shown by the
installer.) This removes all Extraplus files and settings.

## Known issues

- Workbooks store the add-in's path; opening a workbook created on another user's
  machine may show a link prompt or `#NAME?` — not yet tested.

## Troubleshooting

- **"Microsoft Excel is running"**: close every Excel window. If none is open, end
  `EXCEL.EXE` in Task Manager.
- **`#NAME?`** in a cell: Excel did not load the add-in. Check that
  **File > Options > Add-ins > Manage: Excel Add-ins > Go...** has **Extraplus**
  ticked; if it is missing, run `install.bat` again.
- **A cell shows `Error: ...`**: the input is not valid; the message says why.
- **Functions stop working after an Anaconda or Python install**: if the
  environment variables `PYTHONHOME` or `PYTHONPATH` are set for your account,
  the installer warns about it; remove them (Settings > search "environment
  variables") or ask your IT department.
- **Blocked by company policy**: some organisations block macros from add-ins,
  scripts, or programs in your user folder. Ask IT to allow `extraplus.xlam` and
  the Extraplus install folder.
- **Anything else**: run `install.bat` again (it repairs the installation). The
  installer log is `%TEMP%\extraplus-setup.log`; include it when you report a
  problem.
