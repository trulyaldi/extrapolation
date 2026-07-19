# Extrapolation

This project fits asymptotic computational-science data with the established
baseline scan and logarithmically linearized regression method. DuckDB stores
and edits **source scientific data only**. Fits are returned as Python objects
and are never persisted in DuckDB.

## Quick start

From the repository root, install the `src/`-layout package, initialize a local
catalog, and synchronize the bundled CSV/XLS sources:

```bash
python -m pip install -e .
python tools/data_db.py init
python tools/data_db.py sync
```

The editable install makes package-qualified imports work from any directory;
no `PYTHONPATH` or notebook `sys.path` edits are needed.

```python
from extrapolation import fit_dataset

result = fit_dataset(
    "large-dataset/be_1po",
    observable="Energy",
)

print(result.summary())
figure, axes = result.plot()
result.plot_log()
result.plot_profile()
```

`FitResult` exposes the selected data, baseline and uncertainty, confidence
interval, model parameters, log-space R², SSR, residuals, predictions, and the
already-computed baseline scan. Plot methods return Matplotlib figure/axes
objects, do not call `plt.show()`, and never rerun fitting. `result.to_dict()`
and `result.to_json()` serialize the result; `result.to_json("run.json")` also
writes `run.manifest.json` with provenance and effective numerical options.

Lower-level access remains available for specialized work:

```python
from extrapolation.fitting import VarProLinearized
from extrapolation.database import DatasetDatabase
```

Legacy `from extrap import VarProLinearized`, `from database import
DatasetDatabase`, and the DataFrame helpers in `functions.py` remain available.
New code should use package-qualified imports and `fit_dataset()`.

## Dataset resolution and selection

The high-level API obtains the independent-variable column from dataset
metadata. Optional error/reference companions are resolved in this fixed order:

1. explicit `error_dataset=` / `reference_dataset=` arguments;
2. relationship metadata (`related_error_dataset` and
   `related_reference_dataset`, including a `relationships` mapping);
3. exact naming conventions `<directory>-err/<stem>_err` and
   `<directory>-init/<stem>`.

There is no fuzzy matching. A missing optional companion becomes `None`; a
missing explicitly named or metadata relationship is an error. Pass `None` to
disable either association. The relationship names and resolution source are
recorded in `result.metadata`. In the current source collection,
`large-dataset-init` is the historical pre-terminal subset; when automatically
associated, its final row is used only as the optional plotting/reference value.

Rows are stably sorted by the metadata independent column. Bounds are inclusive,
and `n_fit` selects the highest-basis rows after bounds and missing-value
filtering:

```python
result = fit_dataset(
    "large-dataset/be_1po",
    "Energy",
    basis_min=500,
    basis_max=6000,
    n_fit=8,
    missing="drop",       # use "raise" to reject any missing required value
    model="exponential", # default "auto" compares the existing three models
)
```

Explicit arguments override metadata defaults. Validation rejects absent or
nonnumeric columns, nonpositive/duplicate basis values, fewer than three finite
points, constant data, invalid error/reference companions, unsupported methods,
and infeasible solver scans with a dataset-specific error. Effective row IDs,
basis range, dropped rows, relationships, and options are recorded in the
result and export manifest.

Fit every observable through the same canonical single-fit path:

```python
from extrapolation import fit_all_observables

results = fit_all_observables(
    "large-dataset/be_1po",
    observables=["Energy", "MV"],
)
```

## Command line

The installed CLI calls the same public database and fitting APIs:

```bash
extrapolate data init
extrapolate data sync
extrapolate data list
extrapolate data show large-dataset/be_1po
extrapolate data export large-dataset/be_1po exports/be_1po.csv

extrapolate fit large-dataset/be_1po --observable Energy
extrapolate fit large-dataset/be_1po --all
extrapolate fit large-dataset/be_1po --observable Energy \
  --n-fit 8 --json outputs/energy.json --csv outputs/summary.csv \
  --plot-dir outputs/figures
```

Use `extrapolate --db /path/catalog.duckdb ...` for another database. The
portable fallback is `python -m extrapolation.cli ...`. The original focused
data tool remains supported:

```bash
python tools/data_db.py init
python tools/data_db.py sync
python tools/data_db.py list
python tools/data_db.py show large-dataset/be_1po
python tools/data_db.py export large-dataset/be_1po /tmp/be_1po.csv
```

## Reproducible studies

Studies intentionally use a small TOML format rather than a workflow engine:

```toml
study = "be_states"
datasets = ["large-dataset/be_1po", "large-dataset/be_3po"]
observables = ["Energy", "MV"]

[fit]
method = "linearized"
n_fit = 8
compute_uq = true

[outputs]
directory = "outputs/be_states"
summary_csv = "summary.csv"
result_json = true
plots = false
```

```bash
extrapolate run studies/be_states.toml
```

Unknown configuration keys and malformed values are rejected. Relative output
paths are resolved beside the study file. A run writes a CSV summary (unless
disabled), result JSON/manifests by default, and `study.manifest.json` containing
the validated configuration, data hashes, selections, numerical options,
package/schema versions, timestamp, and output files. Nothing is written to
DuckDB by a fit or study.

## Source-data catalog

The catalog uses one logical name per configured source path, lower-cased and
without its suffix (for example `large-dataset/be_1po`). `datasets` stores
provenance and hashes, `dataset_columns` preserves the original ordered schema,
and `dataset_rows`/`dataset_cells` preserve ordered numeric values and missing
cells while reconstructing wide pandas DataFrames.

```python
from extrapolation.database import DatasetDatabase

db = DatasetDatabase()
db.initialize()
db.sync_sources()
db.list_datasets()
bundle = db.load_bundle("large-dataset/be_1po")
data = bundle.observations

db.update_value("large-dataset/be_1po", 0, "Energy", -14.5)
row = db.insert_row("large-dataset/be_1po", {"Basis Size": 6200, "Energy": -14.5})
db.delete_row("large-dataset/be_1po", row)
db.export_dataset("large-dataset/be_1po", "exports/be_1po.csv")
db.execute_readonly_sql(
    "SELECT dataset_name, row_count FROM datasets WHERE data_role = ?",
    ["observation"],
)
```

Writes are transactional and parameterized. Synchronization is
**database-authoritative**: unchanged files are skipped, while changed sources
are reported without overwriting database edits. Deliberate source replacement
requires `python tools/data_db.py sync --replace`; a schema change additionally
requires `--allow-schema-change`. Synchronization does not delete datasets when
a source disappears.

To rebuild a disposable default catalog, remove only its generated ignored file
and synchronize again:

```bash
rm data/database/extrapolation.duckdb
python tools/data_db.py sync
```
