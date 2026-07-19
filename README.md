# Extrapolation source-data catalog

This project fits asymptotic extrapolations from scientific data.  DuckDB now
stores and edits the **input data only**; fitting and uncertainty functions
continue to return ordinary Python objects and do not save fit runs to the
database.

## Data model and synchronization

The catalog records one logical dataset per configured source file, using the
lower-case path without its suffix (for example,
`large-dataset/be_1po`).  It stores provenance and hashes in `datasets`,
original ordered column names in `dataset_columns`, and ordered numeric cells
in `dataset_rows`/`dataset_cells`.  This normalized storage supports the
project's different CSV layouts and missing cells while loading the original
wide pandas DataFrame without one SQL table per file.  Files under
`large-dataset-err` are labelled `uncertainty`; the other configured sources
are `observation` datasets.  Existing reference values remain in the separate
`reference_values` catalog.  `sync` also imports the bundled reference-value
text files without overwriting an existing reference catalog value.

Initialize and import the repository sources:

```bash
python tools/data_db.py sync
python tools/data_db.py list
python tools/data_db.py show large-dataset/be_1po
```

The normal synchronization policy is **database-authoritative**.  An unchanged
file is skipped and a changed source is reported, never silently replacing
database values or manual corrections.  To deliberately replace source values,
use `python tools/data_db.py sync --replace`; a changed column layout also
needs `--allow-schema-change`.  Synchronization never deletes a database
dataset when a source file disappears.

To rebuild a disposable local database, remove only the generated ignored file
and run `sync` again:

```bash
rm data/database/extrapolation.duckdb
python tools/data_db.py sync
```

## Python API

When launched from the repository root, Python supports the following import
directly; no `PYTHONPATH` adjustment is required:

```python
from database import DatasetDatabase

db = DatasetDatabase()
db.initialize()

db.list_datasets()
data = db.load_observations("large-dataset/be_1po")
errors = db.load_dataset("large-dataset-err/be_1po_err")
db.update_value("large-dataset/be_1po", row_index=0, column_name="Energy", value=-14.5)
new_row = db.insert_row("large-dataset/be_1po", {"Basis Size": 6200, "Energy": -14.5})
db.delete_row("large-dataset/be_1po", new_row)
db.export_dataset("large-dataset/be_1po", "exports/be_1po.csv")
db.execute_readonly_sql(
    "SELECT dataset_name, row_count FROM datasets WHERE data_role = ?",
    ["observation"],
)
```

All writes use transactions, validate names and finite numeric values, and use
parameterized SQL.  `insert_row` fills omitted observable values with missing
values.  Use `add_or_replace_dataset(..., replace=True)` for a wholly manual
dataset or explicit DataFrame replacement.

Fitting code consumes pandas data rather than SQL.  The transitional
`functions.load_fit_dataset("large-dataset/be_1po")` (and legacy
`upload_df`) retrieves an imported observation DataFrame through this API.
Pass that DataFrame to `VarProLinearized` or the existing summary/plot helpers.
