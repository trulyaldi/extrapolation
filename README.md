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
