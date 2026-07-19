# AGENTS.md

## Project Purpose

This repository implements asymptotic extrapolation and curve-fitting workflows for computational-science datasets.

The current development priority is to reorganize scientific data scattered across CSV files into a convenient DuckDB-backed data layer.

The database is intended to manage **input scientific data**.

It is not intended to persist extrapolation or fitting results.

---

## Current Task

Refactor the `new_db` branch so that:

1. CSV datasets are imported into and organized through DuckDB.
2. Users can inspect, add, update, and delete scientific data comfortably.
3. Extrapolation code loads data through a small Python database API instead of reading arbitrary CSV files directly.
4. Existing fitting and uncertainty-quantification behavior remains intact.
5. Database result-saving logic is removed.

Work directly on the checked-out `new_db` branch.

Do not create or switch to another branch unless explicitly instructed.

---

## Critical Architectural Decision

The database is the source-data management layer.

The intended flow is:

```text
CSV files
    ↓
database import/synchronization
    ↓
DuckDB scientific-data catalog
    ↓
pandas DataFrame
    ↓
extrapolation and uncertainty analysis
```

Do not implement this flow:

```text
fit data
    ↓
save every fitting result to DuckDB
```

Fitting functions should return ordinary Python results. Saving reports or fitting outputs is outside the database layer unless explicitly requested later.

---

## Required Cleanup

Remove obsolete fitting-result persistence, including any code whose main purpose is to maintain an `extrapolation_results` table.

Search the repository for:

```text
save_result
get_result
list_results
extrapolation_results
database.results
```

Remove or refactor:

* Result-saving modules.
* Result-specific SQL migrations.
* Result-specific indexes.
* Calls to `save_result(...)` from fitting classes or functions.
* Tests that exist only to verify result persistence.
* Dead imports and unused parameters created by this cleanup.

Do not remove fitting calculations, returned fit dictionaries, uncertainty calculations, plots, or report-generation code merely because result persistence is removed.

Before deleting a file, inspect its references across the repository.

---

## Data That Must Be Preserved

The CSV directories contain scientific data and should be treated as valuable source material.

Likely data locations include:

```text
large-dataset/
large-dataset-init/
large-dataset-err/
new-dataset/
small-dataset/
muon/
data/
```

Inspect the actual repository rather than assuming every listed directory exists.

Do not delete CSV files solely because their names contain:

```text
test
init
err
error
```

Some files with names such as `*_test.csv` may contain real scientific observations or uncertainty rows rather than disposable test fixtures.

Determine file semantics from their contents and how the existing code uses them.

Do not alter original CSV files during the initial refactor.

---

## Database Design Goals

Use DuckDB as the persistent store.

The design must support:

* Multiple scientific datasets.
* Different CSV column layouts.
* Dataset names and source-file provenance.
* Ordered independent-variable values.
* Multiple expectation-value or observable columns.
* Missing values.
* Uncertainty or error data.
* Reimporting changed CSV files.
* Querying through SQL when useful.
* Loading fitting-ready pandas DataFrames.
* Safe manual modification of database values.

Prefer a simple design that matches the actual data over an abstract enterprise schema.

Avoid introducing unnecessary concepts such as:

* basis-family fields that are not required by the project;
* user accounts;
* fitting-result history;
* generic workflow engines;
* web APIs;
* ORM frameworks;
* distributed database features.

---

## Schema Guidance

First inspect every distinct CSV shape.

Then choose the simplest schema that handles those shapes without losing information.

A normalized catalog may contain concepts such as:

```text
datasets
source_files
dataset_columns or properties
measurements
uncertainties
```

A wide-table approach may also be appropriate if it makes editing and DataFrame retrieval substantially simpler.

Do not blindly create one SQL table per CSV before examining naming collisions, schema evolution, and query ergonomics.

Do not blindly force all data into long format if the fitting pipeline constantly needs to reconstruct the original wide DataFrames.

Whichever design is selected, document the tradeoff in the repository.

The database must preserve:

* logical dataset identity;
* original source path;
* original column names;
* numeric values;
* missing cells;
* row ordering or independent-variable ordering;
* whether values are observations, uncertainties, references, or metadata.

---

## Import and Synchronization Requirements

Provide one clear entry point for importing the repository’s CSV data.

A command-line interface is preferred, for example:

```bash
python tools/data_db.py sync
```

Equivalent naming is acceptable if it fits the repository.

The import process should:

1. Discover configured CSV files.
2. Identify the logical dataset represented by each file.
3. Validate required columns.
4. Convert valid numeric values safely.
5. Preserve missing values as SQL `NULL` or the corresponding pandas missing value.
6. Import all data in a transaction.
7. Report malformed files clearly.
8. Avoid silently discarding cells.
9. Avoid duplicating unchanged data on repeated runs.
10. make reimport behavior explicit.

Prefer deterministic, idempotent imports.

Using file hashes or modification metadata to skip unchanged sources is acceptable.

Do not silently reinterpret an existing dataset when its schema changes.

---

## Database Modification Requirements

The pipeline must make the database comfortable to modify.

Provide a small, documented Python API for common operations.

The intended style is similar to:

```python
from database import DatasetDatabase

db = DatasetDatabase()

db.list_datasets()
db.load_dataset("large-dataset/be_1po")
db.update_value(...)
db.insert_row(...)
db.delete_row(...)
db.export_dataset(...)
```

The exact class and method names may differ.

At minimum, users must be able to:

* list datasets;
* inspect dataset metadata;
* load a dataset as a pandas DataFrame;
* add or replace a dataset;
* update a value;
* add a row;
* delete a row;
* delete a dataset;
* export a dataset to CSV;
* execute read-only SQL for advanced inspection.

Write operations must use transactions.

Validate dataset names, column names, and numeric values.

Do not build SQL statements by directly interpolating untrusted values.

Identifiers that cannot be parameterized must be strictly validated or safely quoted.

---

## Source Synchronization and Manual Edits

Resolve the interaction between CSV reimport and manual database edits explicitly.

Do not implement a system where a routine synchronization silently destroys database corrections.

Choose and document one of these policies:

### Database-authoritative policy

After initial import, the database becomes authoritative. Reimport is an explicit replacement operation.

### Override policy

Imported source values remain separate from manual overrides, and effective values combine the two.

### Explicit conflict policy

Synchronization detects database modifications and requires an explicit conflict-resolution option.

Prefer the simplest policy that remains safe and understandable.

Include tests for the selected behavior.

---

## Fitting API Integration

The fitting layer should receive pandas DataFrames or NumPy arrays.

Database-specific SQL should remain inside the database package.

Preferred separation:

```text
database package
    loads and modifies data

fitting package
    fits supplied data

application or notebook
    selects a dataset and connects the two
```

Avoid placing SQL queries directly inside numerical fitting classes.

Adapt existing helpers such as `upload_basis` and `upload_error` only when necessary for compatibility.

A transitional wrapper is acceptable:

```python
def upload_basis(dataset_name):
    return database.load_observations(dataset_name)
```

New code should use clearly named database loading methods.

---

## Extrapolation Behavior That Must Remain Unchanged

Preserve the scientific fitting methodology unless a bug is directly encountered.

The project uses a hybrid linear/nonlinear parameter scan:

1. Scan candidate asymptotic baseline values.
2. Subtract each candidate baseline from the dependent variable.
3. Reject candidates that make the logarithm invalid.
4. Apply logarithmic linearization.
5. Run ordinary least-squares regression.
6. Select the baseline according to the configured residual or fit-quality criterion.
7. Quantify uncertainty from degradation of the profile around the optimum.

Do not replace this approach with an unrelated nonlinear optimizer.

Do not change numerical defaults, scan ranges, uncertainty thresholds, or result meanings without documenting the reason and adding targeted tests.

Use vectorized NumPy or SciPy operations where practical.

Handle logarithm-domain errors explicitly.

---

## Repository Inspection Before Editing

Before implementing changes:

1. Run:

```bash
git status --short
git branch --show-current
```

2. Confirm the current branch is:

```text
new_db
```

3. Inspect:

```text
README files
requirements files
pyproject.toml or setup files
src/
sql/
tests/
tools/
notebooks
CSV directories
```

4. Search all references to database modules and result persistence.

5. Inspect representative files from every distinct CSV format.

6. Identify existing test commands before changing code.

Do not assume the ZIP bundle or any prior generated patch is authoritative. Work from the repository’s current contents.

---

## Implementation Order

Use this order unless repository evidence requires a small adjustment:

1. Map CSV formats and existing data-loading behavior.
2. Map current database files, migrations, and APIs.
3. Remove fitting-result persistence.
4. Design the minimum viable source-data schema.
5. Implement database initialization and migration.
6. Implement CSV import and synchronization.
7. Implement query and modification APIs.
8. Integrate DataFrame loading with the fitting pipeline.
9. Add focused tests.
10. Add concise usage documentation.
11. Run validation.
12. Review the final diff for unrelated changes.

Keep commits or logical change groups easy to review.

---

## Testing Requirements

Add focused automated tests for:

* database creation from an empty state;
* migration from the current branch state where practical;
* importing at least one representative CSV shape;
* importing files with missing values;
* importing observations and uncertainty/error data;
* repeated import without duplication;
* loading a fitting-ready DataFrame;
* updating a value;
* inserting and deleting a row;
* persistence of manual edits under the chosen synchronization policy;
* deletion or replacement of a dataset;
* absence of result-saving calls from the fitting path;
* SQL parameter safety for user-provided values.

Use temporary directories and temporary DuckDB files in tests.

Do not write tests against the developer’s real database.

Do not require network access.

Do not duplicate the complete production CSV collection inside test fixtures. Use compact representative fixtures.

---

## Validation Commands

Discover and use the repository’s real commands.

Likely checks may include:

```bash
python -m pytest
python -m compileall src
```

Also run the new import CLI against the repository data when safe.

Example:

```bash
python tools/data_db.py sync
python tools/data_db.py list
```

If a command cannot be run because of a missing dependency or environment issue, report the exact failure.

Do not claim tests passed unless they were executed successfully.

---

## Documentation Requirements

Add concise documentation covering:

* what the database stores;
* what it deliberately does not store;
* schema overview;
* how to initialize the database;
* how to import or synchronize CSV files;
* how to list datasets;
* how to load a pandas DataFrame;
* how to edit data;
* how manual edits interact with CSV reimport;
* how to export data;
* how fitting code accesses database data;
* how to rebuild a disposable local database.

Include runnable command examples.

Do not write extensive generic database tutorials.

---

## Code Quality

Follow the repository’s existing Python style.

Prefer:

* small functions;
* type hints on public APIs;
* explicit transactions;
* context managers;
* `pathlib.Path`;
* parameterized SQL;
* descriptive error messages;
* pandas and NumPy interoperability;
* deterministic behavior;
* narrow module responsibilities.

Avoid:

* hidden global database connections;
* import-time schema mutation;
* mutable default arguments;
* broad `except Exception` blocks without context;
* duplicated SQL across modules;
* implicit deletion during synchronization;
* needless abstraction layers.

---

## Scope Control

Do not:

* redesign the mathematical extrapolation algorithm;
* rename every existing module;
* rewrite all notebooks;
* introduce a GUI;
* introduce a server;
* add cloud database infrastructure;
* persist every fitting run;
* modify scientific input data without explicit justification;
* delete files based only on their names;
* perform unrelated formatting across the repository.

Make the smallest coherent refactor that establishes the new database direction.

---

## Git Safety

Do not discard existing uncommitted user changes.

Do not run:

```bash
git reset --hard
git clean -fd
git checkout -- .
```

Do not force-push.

Do not amend existing commits unless explicitly asked.

Do not commit generated database files unless the repository already intentionally tracks them.

Generated local files such as these should normally be ignored:

```text
*.duckdb
*.duckdb.wal
*.db
__pycache__/
.pytest_cache/
```

Check existing ignore rules before changing `.gitignore`.

---

## Final Deliverable

At completion, provide:

1. A concise explanation of the selected database design.
2. A list of deleted result-persistence components.
3. A list of major files added or modified.
4. Instructions for importing existing CSV data.
5. Examples for listing, loading, modifying, and exporting datasets.
6. The selected CSV/manual-edit synchronization policy.
7. Tests and validation commands executed.
8. Any remaining limitations or ambiguous source-data formats.
9. `git status --short`.
10. A suggested commit message.

Do not merely produce a design document. Implement the refactor in the repository.

Do not report success until the resulting code has been inspected and the available tests have been run.
