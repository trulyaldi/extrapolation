# extrapolation

Extrapolation of basis-set sequences to the complete-basis limit. The repository has three parts:

- **Library** (`src/extraplus/`): the `extraplus` class, installable as a Python package.

  ```bash
  pip install -e .            # library only
  pip install -e ".[excel]"   # plus xlwings for the Excel UDFs
  pip install -e ".[dev]"     # plus pytest and jupyter
  ```

  ```python
  from extraplus import extraplus
  ```

- **Excel** (`excel/`): xlwings UDF wrappers (`extrapolation.py`) around `extraplus`, shipped as the `extraplus.xlam` add-in with a one-click Windows installer. User guide: [excel/README.md](excel/README.md); building a release: [excel/build/README.md](excel/build/README.md).

- **Research** (`research/`): notebooks, plotting helpers (`notebooks/functions.py`), and the datasets under `data/`.

`legacy/` holds the earlier `extrapolator.py`, kept for reference.

## Python

Requires Python 3.11 or later. Install the wheel from a release that includes this API
(or build it with `uv build --wheel`, which writes it to `dist/`):

```bash
pip install extraplus-<version>-py3-none-any.whl
```

```python
from extraplus import compare, extrapolate

basis = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
energy = [-14.6661876127332, -14.6667402339119, -14.6672828224583, -14.6673249180953,
          -14.66734057696, -14.6673445259951, -14.6673478029866, -14.6673503395393,
          -14.6673519816691, -14.6673531093866, -14.6673536576132, -14.6673541629258]

result = extrapolate(basis, energy, model="exponential")   # or model=1, 2, 3 as in Excel
print(result.summary())
print(result.value, result.uncertainty)

fig = result.plot()                  # a Matplotlib figure; also result.plot_linearized()
fig.savefig("be_1se_fit.png", dpi=150)

print(compare(basis, energy))        # all three models, one row each (a pandas DataFrame)
```

`extrapolate()` also takes `b_fixed` (fix the decay rate B) and `n_fit` (fit only the first
`n_fit` points). Invalid input raises `ValueError` with the same message the Excel functions
show after "Error: ". The Excel functions call this same API, so both give the same results.
