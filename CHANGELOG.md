# Changelog

## Unreleased

### Added

- Python API, installable from the wheel (`pip install extraplus-<version>-py3-none-any.whl`):
  - `extraplus.extrapolate(x, y, model="exponential", b_fixed=None, n_fit=None)` returns a
    `Result`; `model` is a name or 1, 2, 3 as in Excel; `x`/`y` are any array-like.
  - `extraplus.compare(x, y, b_fixed=None, n_fit=None)` returns a pandas DataFrame with one
    row per model.
  - `extraplus.Result` (frozen dataclass): `value`, `uncertainty`, `decay_rate`, `amplitude`,
    `r2`, `model`, `n_points`, `n_fit_used`, `x_max`, `b_fixed`, and `summary()`, `to_dict()`,
    `plot()`, `plot_linearized()`.
  - `extraplus.__version__`; the package is typed (`py.typed`).
  - The API never prints; a non-finite uncertainty gives a `RuntimeWarning`.
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
  (`excel/build/build_addin.sh`) before the next release.
