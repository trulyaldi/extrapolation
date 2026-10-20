# Changelog

## 0.4.0 (unreleased)

Makes large Excel workbooks robust: results no longer get lost when a workbook is opened
on another computer, recalculation is cached, and finished sheets can be turned into plain
numbers. The Python API and the numeric results are unchanged.

### Added

- Excel function `EXTRAP_ALL_V(x, y, [b_fixed])`: all three models in one call, a 3 × 3
  array with rows C, uncertainty, B and columns models 1, 2, 3; column n is exactly
  `EXTRAPn_V`. `b_fixed` is blank or 3 cells (one B per model, a blank cell fits that B),
  for example the B row of another `EXTRAP_ALL_V`. No `n_fit` argument. It does not resize
  itself: it spills in dynamic-array Excel, or is entered over 3 × 3 cells with
  Ctrl+Shift+Enter.
- Result cache for the Excel functions: every fit is kept in memory and in
  `<install folder>\resultcache\results.sqlite3`, shared by all Excel instances. A result is
  reused only for exactly the same cell values (compared bit for bit, by type) and the same
  extraplus code and library versions, so outputs are identical with and without it. At most
  20,000 results in memory and 100,000 in the file (oldest first out). Failed fits are never
  stored. A locked, damaged or unwritable file is bypassed (a damaged one is renamed
  `*.corrupt-<time>` and replaced); a function never fails because of the cache.
  - Switch: environment variable `EXTRAPLUS_CACHE` = `off` / `memory` / `on` (default);
    `EXTRAPLUS_CACHE_DIR` moves the file.
  - `EXTRAPLUS_CACHE_STATS()` shows the mode, the file and the hit/miss counts.
- Ribbon tab **Extraplus > Write values**: replaces the Extraplus formulas of the selection
  (or of the sheet) by the numbers they compute, in one Python pass. Chained `b_fixed`
  references to other Extraplus formulas are resolved in the same pass. A copy of the
  workbook is saved next to it first, and each replaced formula (text, version, inputs
  digest) is listed on the sheet "Extraplus values". Formulas that give an error, nested or
  otherwise unrecognised formulas, and formulas depending on a formula that was not
  replaced are kept and listed.
- On opening a workbook, the add-in repoints its links to `extraplus.xlam` at another path
  (a workbook saved on another computer or by another user) to the add-in loaded in this
  Excel.
- Before saving, the add-in warns if cells with Extraplus formulas show `#NAME?`, `#REF!` or
  `#VALUE!`, so a failed recalculation is not saved over the stored numbers.

### Changed (failure path only)

- `EXTRAP1`–`3_V` and `EXTRAP1`–`3_H` show their "Error: ..." text in each of their 3 cells.
  Before, they returned a single text, and xlwings' resizing (`expand="table"`) then cleared
  the 3-cell array and re-entered the formula as one cell; the emptied B cell made every
  formula using it as `b_fixed` silently fit B instead. Side effect: a `_V`/`_H` formula that
  occupies one cell (just typed, or shrunk by the old behaviour) now also resizes itself to 3
  cells when it shows an error, as it already did on success, writing over the 2 cells next
  to it.
- An Excel error value (`#N/A`, `#NAME?`, ...) in the `b_fixed` or `n_fit` argument of the
  numeric functions now gives "Error: b_fixed must be a number." / "Error: n_fit must be a
  number.". Before, it was read as a blank cell, so B was fitted / all points were used.
- Successful results of every existing function are unchanged (bit for bit), and no
  function was renamed or changed its arguments. The `_PLOT` functions are unchanged.

### Upgrading

Close Excel and run `install.bat` from the 0.4.0 zip (it updates an existing install in
place). Existing workbooks keep working: they call the same functions, and links to an
add-in at another path are repointed when they are opened. Maintainers: the add-in must be
rebuilt (`excel/build/build_addin.sh`) after the version bump, because it now contains the
relink, save-guard and ribbon code and the new functions.

## 0.3.0 (2026-09-24)

### Added

- Python API, installable from the wheel (`pip install extraplus-<version>-py3-none-any.whl`):
  - `extraplus.extrapolate(x, y, model="exponential", b_fixed=None, n_fit=None)` returns a
    `Result`; `model` is a name or 1, 2, 3 as in Excel; `x`/`y` are any array-like.
  - `extraplus.compare(x, y, b_fixed=None, n_fit=None)` returns a pandas DataFrame with one
    row per model.
  - `extraplus.Result` (frozen dataclass): `value`, `uncertainty`, `decay_rate`, `amplitude`,
    `r2`, `model`, `n_points`, `n_fit_used`, `x_max`, `b_fixed`, and `summary()`, `to_dict()`,
    `predict()`, `plot()`, `plot_linearized()`.
  - `Result.predict(x)`: the fitted curve A·φ(x/x_max) + C at a finite basis size x, in the
    units of y (a float, or an array for array-like x). `x` must be a finite number > 0;
    otherwise it raises `ValueError` ("target is required.", "target must be a number.",
    "target must be finite." or "target must be greater than zero.").
  - `extraplus.__version__`; the package is typed (`py.typed`).
  - The API never prints; a non-finite uncertainty gives a `RuntimeWarning`.
- Excel functions `FINITE_EXTRAP1(x, y, target, [b_fixed], [n_fit])`, `FINITE_EXTRAP2` and
  `FINITE_EXTRAP3`: the fitted value at the finite basis size `target` (`Result.predict(target)`),
  with the same input handling as `EXTRAP1`-`3`. A missing or invalid target shows
  "Error: <message>".
- The `extraplus` research class stays importable as before (`from extraplus import extraplus`).

### Changed

- The Excel functions are now thin adapters over the Python API: input checks, fitting and
  plots live in the package, so Excel and Python give the same results by construction.
  Numeric results and plots are unchanged.
- Error messages (Excel shows them after "Error: "; Python raises them as `ValueError`):

  | Old | New |
  |---|---|
  | `x_range and y_range must contain the same number of cells.` | `x and y must have the same length.` |
  | `x_range must contain at least two distinct numeric values.` | `x must contain at least two distinct values.` |
  | `y_range is constant, so an extrapolation cannot be fitted.` | `y is constant, so an extrapolation cannot be fitted.` |
  | `could not convert string to float: '<text>'` (b_fixed) | `b_fixed must be a number.` |
  | `TypeError: float() argument must be a string or a real number, not '<type>'` (b_fixed) | `b_fixed must be a number.` |
  | `could not convert string to float: '<text>'` (n_fit) | `n_fit must be a number.` |
  | `OverflowError: cannot convert float infinity to integer` (n_fit) | `n_fit must be finite.` |
  | `cannot convert float NaN to integer` (n_fit) | `n_fit must be finite.` |

  New (Python only): `model must be 'exponential', 'sqrt_exponential', 'power_law', 1, 2 or 3.`

- `excel/extrapolation.py` changed, so the Excel add-in must be rebuilt
  (`excel/build/build_addin.sh`) before the next release. `FINITE_EXTRAP1`-`3` appear in Excel
  only after that rebuild.
