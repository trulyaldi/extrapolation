# Extraplus for Excel

THIS MARKDOWN FILE CONTAINS ALL THE NECESSARY INFORMATION. please, read it carefully. 

Extraplus adds extrapolation functions to Excel on Windows:

```
=EXTRAP1(A2:A20, B2:B20)
```

The functions work in any workbook (`.xlsx` or `.xlsm`), saved in any folder.
You don't need Python, admin rights, or any Excel settings. Extraplus needs
64-bit Excel.

## Install

1. Close Excel.
2. Unzip `extraplus-excel-vX.Y.Z.zip` (right-click > Extract All).
3. Double-click `install.bat`. If Windows says "Windows protected your PC",
   click **More info > Run anyway**.
4. Wait for `SUCCESS` (a few minutes the first time), then press Enter.

You can delete the unzipped folder afterwards. Excel takes a few seconds
longer to start, because the add-in loads Python with it.

## Functions

`x` is the range of basis sizes and `y` the range of values. The digit selects
the model: **1** exponential, **2** square-root exponential, **3** power law.

| Function | Returns |
|---|---|
| `EXTRAP1(x, y, [b_fixed], [n_fit])` | extrapolated value C at infinite basis |
| `FINITE_EXTRAP1(x, y, target, [b_fixed], [n_fit])` | the fitted curve's value at the finite basis size `target`, usually one beyond your data. Example: with x = 10000 to 14000 in `A2:A4`, `=FINITE_EXTRAP1(A2:A4, B2:B4, 15000)` gives the value expected at basis size 15000 |
| `UNCERTAINTY1(x, y, [b_fixed], [n_fit])` | uncertainty of C |
| `DECAY_RATE1(x, y, [b_fixed], [n_fit])` | decay parameter B |
| `EXTRAP1_V(...)` / `EXTRAP1_H(...)` | C, uncertainty and B in 3 cells, down / across |
| `EXTRAP_ALL_V(x, y, [b_fixed])` | all three models at once, 3 × 3 cells: rows C, uncertainty, B; columns models 1, 2, 3 (column 1 is `EXTRAP1_V`, and so on) |
| `EXTRAP1_PLOT(...)` | plot of the data, fit and extrapolated value |
| `EXTRAP1_PLOT_LOG(...)` | plot of the fit in linearized form |
| `EXTRAPLUS_VERSION()` | installed version |
| `EXTRAPLUS_CACHE_STATS()` | state of the result cache (see below) |

Every function except `EXTRAP_ALL_V` also exists for models 2 and 3 (`EXTRAP2`, `EXTRAP3`, ...).

Optional arguments:
- `b_fixed`: use this value for B instead of fitting it. For `EXTRAP_ALL_V`,
  three cells with one B per model, for example the B row of another
  `EXTRAP_ALL_V`: `=EXTRAP_ALL_V(A2:A20, C2:C20, E4:G4)`. A blank cell
  among the three fits that model's B.
- `n_fit`: fit only the first `n_fit` points. Useful for checking whether the
  extrapolation is stable as more points are added.

`EXTRAP_ALL_V` replaces three `EXTRAPn_V` formulas with one. Select 3 × 3
cells, type the formula and press Ctrl+Shift+Enter (Excel with dynamic arrays
also fills the 3 × 3 cells from a single cell).

`EXTRAP1_V`, `EXTRAP1_H` and `EXTRAP_ALL_V` fill the cells they are entered
in, like any Excel array function: entered in more cells than they return,
the extra cells show `#N/A`.

If the input is invalid, the cell shows a message such as
`Error: At least three paired numeric x/y observations are required.`
(`EXTRAP1_V` and the others show it in each of their cells). A `b_fixed` or
`n_fit` cell that shows an Excel error such as `#N/A` gives
`Error: b_fixed must be a number.`; only a blank cell means "fit B".

## Big workbooks

### Faster recalculation: the result cache

Each fit is remembered, so recalculating a sheet (after inserting a column,
reopening the file, pressing F9) does not fit unchanged data again. The
cache is kept in memory while Excel runs and in the file
`resultcache\results.sqlite3` in the install folder (usually
`%LOCALAPPDATA%\Extraplus`), which is meant to be shared by all Excel
windows on this PC and survives restarts. A result is reused
only for exactly the same input values and the same Extraplus version, so
the numbers are identical to computing them again. The file holds at most
100,000 results (the oldest are removed first) and can be deleted at any
time when Excel is closed.

To turn the cache off, close Excel, open a Command Prompt and run
`setx EXTRAPLUS_CACHE off` (or `setx EXTRAPLUS_CACHE memory` to keep it only
in memory), then start Excel again. `setx EXTRAPLUS_CACHE on` turns it back
on. `=EXTRAPLUS_CACHE_STATS()` shows the current state (press Ctrl+Alt+F9 to
refresh it).

### Finished sheets: Extraplus > Write values

A sheet with thousands of Extraplus formulas recalculates all of them
whenever Excel decides to (opening the file in another Excel version,
inserting columns, updating links). When the results are final, replace the
formulas with their numbers:

1. Save the workbook.
2. Select the cells (or one cell for the whole sheet) and click
   **Extraplus > Write values** on the ribbon.

Each formula is computed again from its input cells and replaced by its
numbers. Formulas that give an error, formulas inside other formulas, and
formulas whose `b_fixed` comes from a formula that was not replaced are kept
as they are and listed. Before anything is replaced, a copy of the workbook
is saved next to it (`<name> (before Extraplus values <date time>).xlsm`),
and every replaced formula is listed on the sheet **Extraplus values** with
the version and a fingerprint of its inputs. To get the formulas back, open
that copy, or type the formula from the list again.

### Workbooks made with Extraplus 0.4.0 or earlier

Earlier versions used an add-in file, `extraplus.xlam`, and their formulas
refer to it by its location, for example
`='C:\Users\Anna\AppData\Local\Extraplus\extraplus.xlam'!EXTRAP1_V(A2:A6,B2:B6)`.
When you open such a workbook, Extraplus offers to convert it: it removes that
reference, so the formula becomes `=EXTRAP1_V(A2:A6,B2:B6)`, and removes the
link to the old file. Nothing else in the formulas changes, and the numbers
are computed by the same functions. Save the workbook afterwards to keep the
change. You can also convert later with **Extraplus > Convert old formulas**.

Converted workbooks no longer depend on where Extraplus is installed, so they
work on any computer with Extraplus.

### Saving a workbook with errors

Before you save a workbook in which cells with Extraplus formulas show
`#NAME?`, `#REF!` or `#VALUE!`, Extraplus asks whether you really want to
save: saving would replace the numbers stored in the file with those errors.
If the add-in is not installed at all, none of this happens: don't save
such a workbook; install Extraplus first.

## Not yet tested in Excel (0.4.0)

The result cache, `EXTRAP_ALL_V`, **Write values** and the error shapes are
tested in Python. These parts run inside Excel and have **not** been tested
in a real Excel yet: repointing links when a workbook is opened (and whether
Excel recalculates while doing so), the warning before saving, the ribbon
button, two Excel windows sharing the cache file, and entering `EXTRAP_ALL_V`
from a single cell. Test them on copies of your workbooks first; the checklist
is in `excel/build/manual-tests-0.4.0.md` in the source repository.

## Update

Close Excel and run `install.bat` from the new zip. Your workbooks are not
affected.

## Uninstall

Close Excel, then go to **Settings > Apps > Extraplus for Excel > Uninstall**.

## Troubleshooting

- **"Microsoft Excel is running"**: close all Excel windows and try again.
- **`#NAME?` in a cell**: the add-in isn't loaded. Check that **extraplus**
  (`extraplus.xll`) is ticked under **File > Options > Add-ins > Manage: Excel
  Add-ins > Go**. If it's missing, run `install.bat` again. A formula that
  still starts with `'...\extraplus.xlam'!` was made with an earlier version:
  use **Extraplus > Convert old formulas**.
- **Anything else**: run `install.bat` again, which repairs the installation.
  If that doesn't help, send me the log files `%TEMP%\extraplus-setup.log` and
  `extraplus.log` from the install folder (usually `%LOCALAPPDATA%\Extraplus`).
