# extraplus

extraplus extrapolates a sequence of results computed with growing basis size, such as
energies, to the infinite-basis limit, with an uncertainty for that limit. It fits one of three
models (exponential, square-root-exponential, power law) and can be used from Python or from
Excel.

## Installation

Requires Python 3.11 or later. A virtual environment is recommended:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

Then install the wheel file (keep its name as downloaded):

```bash
pip install extraplus-<version>-py3-none-any.whl
python -c "import extraplus as ep; print(ep.__version__)"
```

This also installs numpy, scipy, pandas and matplotlib.

## Quick start

```python
from extraplus import compare, extrapolate

basis = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
energy = [-14.6661876127332, -14.6667402339119, -14.6672828224583, -14.6673249180953,
          -14.66734057696, -14.6673445259951, -14.6673478029866, -14.6673503395393,
          -14.6673519816691, -14.6673531093866, -14.6673536576132, -14.6673541629258]

res = extrapolate(basis, energy, model="exponential")   # or model=1, 2, 3 as in Excel
print(res.summary())
print(res.value, res.uncertainty)
print(res.predict(1500))             # the fitted curve at a larger basis, x = 1500

fig = res.plot()                     # a Matplotlib figure
fig.savefig("be_1se_fit.png", dpi=150)

print(compare(basis, energy))        # all three models, one row each
```

## API reference

### `extrapolate(x, y, model="exponential", b_fixed=None, n_fit=None) -> Result`

| Parameter | Default | Meaning |
|---|---|---|
| `x`, `y` | | Basis sizes and values, in the same order: list, tuple, NumPy array or pandas Series. Pairs where x or y is missing (`None`), text or not finite are skipped; at least three usable pairs are needed. |
| `model` | `"exponential"` | `"exponential"`, `"sqrt_exponential"`, `"power_law"`, or `1`, `2`, `3` (see [Models](#models)). |
| `b_fixed` | `None` | Fix B at this positive value instead of fitting it. |
| `n_fit` | `None` | Fit only the first `n_fit` usable pairs (at least 3; truncated to a whole number; larger values mean all pairs). |

### `compare(x, y, b_fixed=None, n_fit=None) -> pandas.DataFrame`

Runs `extrapolate()` with all three models, with the same `x`, `y`, `b_fixed` and `n_fit`.
Returns one row per model, indexed by model name, with the columns `value`, `uncertainty`,
`decay_rate`, `amplitude`, `r2`, `n_points`, `n_fit_used`, `x_max` and `b_fixed`.

### `Result` attributes

`Result` is a frozen dataclass. Values are in the units of y unless noted.

| Attribute | Type | Meaning | Excel |
|---|---|---|---|
| `value` | float | C, the extrapolated limit for x → ∞ | `EXTRAP1`–`3` |
| `uncertainty` | float | uncertainty of C; `inf` when the fit does not constrain C | `UNCERTAINTY1`–`3` |
| `decay_rate` | float | B, fitted or equal to `b_fixed` (for t = x/x_max) | `DECAY_RATE1`–`3` |
| `amplitude` | float | A; positive for a decreasing sequence, negative for an increasing one | |
| `r2` | float | R² of the fit in linearized (log) space | |
| `model` | str | `"exponential"`, `"sqrt_exponential"` or `"power_law"` | |
| `n_points` | int | usable (x, y) pairs in the input | |
| `n_fit_used` | int | pairs that were fitted (the first `n_fit_used`) | |
| `x_max` | float | largest x of all usable pairs | |
| `b_fixed` | float or None | the fixed B, or `None` when B was fitted | |

`EXTRAP1_V`–`3_V` and `EXTRAP1_H`–`3_H` return `value`, `uncertainty` and `decay_rate`
together; `EXTRAP_ALL_V` returns them for all three models (column n is `EXTRAPn_V`);
`EXTRAPLUS_VERSION()` returns `extraplus.__version__`.

### `Result` methods

| Method | Returns | Excel |
|---|---|---|
| `summary()` | a readable multi-line `str` (printing it is up to you) | |
| `to_dict()` | a `dict` of the ten attributes above, as plain Python values | |
| `predict(x)` | the fitted curve A·φ(x/x_max) + C at the basis size `x`, in the units of y: a `float` for a scalar, an array of the same shape for array-like `x`. Meant for `x` beyond the data; `x` must be a finite number > 0, otherwise `ValueError` | `FINITE_EXTRAP1`–`3` |
| `plot()` | a Matplotlib `Figure`: all points (those beyond `n_fit` as open markers), the fitted curve to 1.5 x_max, and C ± uncertainty | `EXTRAP1_PLOT`–`3_PLOT` |
| `plot_linearized()` | a Matplotlib `Figure`: ln\|y − C\| against the model's transformed x, with the fit and its 95% confidence band | `EXTRAP1_PLOT_LOG`–`3_PLOT_LOG` |

The figures are open pyplot figures: show or save them, then close them with
`matplotlib.pyplot.close(fig)`.

### Errors and warnings

Input that cannot be fitted raises `ValueError`. The messages are the ones the Excel functions
show after "Error: ":

- `x and y must have the same length.`
- `At least three paired numeric x/y observations are required.`
- `x must contain at least two distinct values.`
- `The largest x value cannot be zero because x is scaled by x_max.`
- `y is constant, so an extrapolation cannot be fitted.`
- `b_fixed must be a number.` / `b_fixed must be finite.` / `b_fixed must be greater than zero.`
- `n_fit must be a number.` / `n_fit must be finite.` / `n_fit must be at least 3.`
- `model must be 'exponential', 'sqrt_exponential', 'power_law', 1, 2 or 3.`
- From `predict()` (`FINITE_EXTRAP1`–`3`): `target is required.` / `target must be a number.` /
  `target must be finite.` / `target must be greater than zero.`

When the uncertainty is not finite, `extrapolate()` and `compare()` issue a `RuntimeWarning`
and still return the result. The functions never print.

## Models

Each model is `y = A·φ(t) + C` with `t = x / x_max`:

| Number | Name | Fitted model |
|---|---|---|
| 1 | `exponential` | `y = A·exp(−B·t) + C` |
| 2 | `sqrt_exponential` | `y = A·exp(−B·√t) + C` |
| 3 | `power_law` | `y = A·t^(−B) + C` |

C is the limit of y for x → ∞ (the extrapolated value), A the amplitude of the part that
vanishes, and B the rate at which it vanishes in t. The fit works in linearized space,
ln|y − C| = ln|A| − B·τ with τ = t, √t or ln t, and chooses C to maximize R² there.
`x_max` is the largest x of all usable points, also when only the first `n_fit` are fitted.

## Advanced: the `extraplus` class

`extrapolate()` is built on the research engine, the `extraplus` class (`from extraplus import
extraplus`), which remains available for research use. It takes a pandas DataFrame
(`extraplus(df, x_col, y_col, err_df=None, inf_df=None, b_init=None, n_fit=None,
use_energy_b=True)`) and fits several models in one run with `fit_linearized()`. `inf_df` supplies
a known limit (the last row of its `y_col` column), and `err_df` its error, which `plot()` draws
for comparison. `use_energy_b` (on by default) takes B from a fit of an `"Energy"` column of the
same DataFrame when `y_col` is another property (halved for singular properties). The class
prints progress and result tables, and works in scaled y units internally.

## Excel

The same functions are available in Excel through an add-in with a one-click Windows installer;
see [excel/README.md](excel/README.md).

## Development

The repository contains the library (`src/extraplus/`), the Excel add-in (`excel/`), research
notebooks and datasets (`research/`), and the earlier `extrapolator.py` (`legacy/`, kept for
reference).

```bash
pip install -e ".[dev]"      # editable install with pytest and jupyter
pip install -e ".[excel]"    # plus xlwings, for the Excel functions
pytest                       # from the repository root
```

Building the Excel add-in and making a release: [excel/build/README.md](excel/build/README.md).
