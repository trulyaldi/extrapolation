## Project Purpose

This repository implements asymptotic extrapolation and uncertainty-analysis workflows for computational-science datasets.

The source-data layer is already organized through DuckDB. The current development priority is to make the fitting workflow:

- low friction for interactive use;
- reusable across notebooks, scripts, and batch studies;
- reproducible;
- easy to validate and test;
- compatible with the existing scientific methodology.

The intended user experience is:

```python
from extrapolation import fit_dataset

result = fit_dataset(
    "large-dataset/be_1po",
    observable="Energy",
)

result.summary()
result.plot()
result.plot_log()
result.plot_profile()
```

Do not redesign the scientific extrapolation algorithm unless a specific, tested bug requires it.

---

## Current Architecture

The intended data flow is:

```text
CSV/XLS source files
        ↓
DatasetDatabase synchronization
        ↓
DuckDB scientific-data catalog
        ↓
pandas DataFrame
        ↓
validated fit selection
        ↓
VarProLinearized numerical engine
        ↓
FitResult
        ↓
plots, summaries, JSON/CSV reports
```

The database stores source scientific data. It must not automatically persist every fitting result.

Fitting results should be ordinary Python objects. Optional exported reports, manifests, figures, JSON, or CSV files may be written explicitly by the user.

---

## Existing Database Layer

Preserve the existing `DatasetDatabase` behavior unless a directly relevant bug is found.

The database layer already supports:

- initialization and migrations;
- source synchronization;
- dataset listing;
- DataFrame loading;
- value and row modification;
- export;
- read-only SQL;
- database-authoritative synchronization safeguards.

Do not redesign the schema as part of the fitting-workflow work.

Do not delete or rewrite existing CSV/XLS source files.

Do not reintroduce fitting-result persistence into DuckDB.

---

## Primary Goal

Create one clear, supported fitting workflow that works consistently from:

- Python scripts;
- Jupyter notebooks;
- command-line tools;
- batch studies;
- tests.

The fitting API must remove routine boilerplate while preserving access to the lower-level solver for advanced users.

Target high-level API:

```python
from extrapolation import fit_dataset

result = fit_dataset(
    dataset="large-dataset/be_1po",
    observable="Energy",
)
```

Target project API, only if it remains a thin convenience layer:

```python
from extrapolation import ExtrapolationProject

project = ExtrapolationProject.open()

result = project.fit(
    dataset="large-dataset/be_1po",
    observable="Energy",
)
```

Do not create multiple competing high-level APIs. Prefer one canonical function, with any project object delegating to it.

---

## Required Work

### 1. Make the repository installable

Add or complete `pyproject.toml` using the `src/` layout.

The project must support:

```bash
python -m pip install -e .
```

After installation, imports must work from any directory without modifying `sys.path`.

Preferred import style:

```python
from extrapolation import fit_dataset, FitResult
from extrapolation.database import DatasetDatabase
from extrapolation.fitting import VarProLinearized
```

Avoid requiring:

```python
from functions import VarProLinearized
```

Maintain compatibility aliases where practical, but new code and documentation should use package-qualified imports.

Do not rename every file at once if a smaller compatibility-preserving package wrapper is sufficient.

### 2. Implement one canonical `fit_dataset()` function

Add a high-level function whose responsibility is orchestration, not numerical reinvention.

Suggested signature:

```python
def fit_dataset(
    dataset: str,
    observable: str,
    *,
    db: DatasetDatabase | None = None,
    method: str = "linearized",
    model: str = "auto",
    n_fit: int | None = None,
    basis_min: float | None = None,
    basis_max: float | None = None,
    use_energy_b: bool | None = None,
    compute_uq: bool = True,
    missing: str = "drop",
    verbose: bool = False,
) -> FitResult:
    ...
```

The exact signature may be adjusted to match actual solver capabilities, but it must remain small and understandable.

The function should:

1. create or reuse `DatasetDatabase`;
2. load the named dataset;
3. obtain dataset metadata;
4. determine the independent-variable column;
5. resolve related error/reference datasets when available;
6. validate the requested observable and selected rows;
7. apply explicit, documented row selection;
8. instantiate the existing numerical solver;
9. run the requested existing fitting method;
10. return a stable `FitResult`.

Do not duplicate the numerical fitting algorithm inside `fit_dataset()`.

Do not put SQL inside numerical fitting classes.

### 3. Add a stable `FitResult`

Create a result type that separates reusable scientific results from solver implementation state.

A dataclass is preferred.

The result should expose, where available:

```python
result.dataset_name
result.observable
result.x_col
result.method
result.model
result.baseline
result.baseline_uncertainty
result.confidence_interval
result.parameters
result.r_squared
result.residual_sum_squares
result.residuals
result.scan_values
result.scan_scores
result.selected_data
result.metadata
```

Use names that match the actual scientific meaning in the repository.

Recommended methods:

```python
result.summary()
result.to_dict()
result.to_json(path)
result.plot()
result.plot_log()
result.plot_profile()
```

Plot methods may delegate to standalone plotting functions.

Do not make plotting required for fitting.

Do not require a live solver object to serialize the scientific result.

If retaining the solver internally is useful for compatibility, keep it optional and excluded from serialization.

### 4. Add dataset bundling and relationship resolution

Remove the need for users to manually load observation, error, and reference DataFrames.

Add a small bundle type or equivalent loader:

```python
bundle = db.load_bundle("large-dataset/be_1po")

bundle.observations
bundle.errors
bundle.reference
bundle.metadata
bundle.x_col
```

If the existing naming conventions can reliably associate files such as:

```text
large-dataset/be_1po
large-dataset-err/be_1po_err
large-dataset-init/be_1po
```

implement that resolution in one documented location.

Prefer explicit metadata relationships when available.

Naming-convention fallback is acceptable, but it must be deterministic and tested.

A missing related error/reference dataset is not automatically an error. Return `None` when optional data is unavailable.

Do not guess between multiple ambiguous matches. Raise a clear error or require explicit selection.

### 5. Add metadata-driven defaults

Use dataset metadata for stable defaults such as:

- independent-variable column;
- observable columns;
- ignored columns;
- default fitting method;
- default fit-point count;
- minimum basis size;
- optional related error dataset;
- optional related reference dataset.

Do not hard-code repository-wide assumptions such as `"basis size"` when metadata already provides the correct column.

Explicit user arguments must override metadata defaults.

Fallback behavior must be documented and deterministic.

### 6. Add pre-fit validation

Validate data before entering logarithms, regressions, or uncertainty scans.

Validation should check, as applicable:

- dataset exists;
- independent-variable column exists;
- observable exists;
- selected columns are numeric;
- enough finite points remain;
- independent-variable values are valid;
- duplicate independent-variable values are handled explicitly;
- row ordering is deterministic;
- error values are finite and positive;
- observation and error rows align;
- reference data has the expected shape;
- the logarithmic transformation has a feasible domain;
- the scan range is nonempty;
- the requested method is supported.

Errors must be actionable.

Prefer:

```text
Cannot fit observable "Energy" in "large-dataset/be_1po":
only 2 finite points remain after filtering; at least 3 are required.
```

Avoid exposing raw NumPy warnings as the primary user-facing failure.

Provide either:

```python
validate_fit_request(...)
```

or validation integrated into `fit_dataset()` with a reusable report type.

### 7. Make row selection explicit and reproducible

Any behavior currently hidden in helpers, including minimum basis size or taking the last N rows, must become explicit configuration.

Support a small set of clear options such as:

```python
fit_dataset(
    dataset,
    observable,
    basis_min=99,
    basis_max=None,
    n_fit=6,
)
```

Record the effective selection in `FitResult.metadata`.

Do not silently reorder or drop rows without recording what happened.

Default sorting by the independent variable is acceptable if documented.

For missing values, support one documented policy such as:

```python
missing="drop"
```

and reject unsupported policies clearly.

### 8. Unify single and batch fitting

All higher-level fitting paths must delegate to the same canonical single-fit implementation.

Implement, or refactor existing helpers toward:

```python
fit_dataset(...)
fit_all_observables(...)
fit_study(...)
```

Example:

```python
results = fit_all_observables(
    "large-dataset/be_1po",
)
```

Batch fitting must not contain a second copy of the numerical algorithm.

Existing helpers such as `fit_all_log`, `fit_all_irls`, or plotting/report classes may remain as compatibility wrappers, but their fitting work should eventually delegate to `fit_dataset()`.

Add regression tests proving that direct low-level fitting and high-level fitting agree numerically for representative data.

### 9. Separate fitting from plotting

Numerical fitting must be usable in headless environments.

Recommended separation:

```python
result = fit_dataset(...)
plot_fit(result)
plot_log(result)
plot_profile(result)
```

Convenience methods on `FitResult` may call these functions.

Plot functions should:

- return matplotlib `Figure` and `Axes` objects;
- not call `plt.show()` unconditionally;
- accept an optional output path;
- not rerun the fit;
- use data stored in `FitResult`;
- preserve current scientific plot meanings.

Do not globally mutate matplotlib style at import time.

### 10. Add a fitting CLI

Add a command-line entry point.

Preferred installed command:

```bash
extrapolate fit large-dataset/be_1po --observable Energy
```

A module or script fallback is acceptable:

```bash
python -m extrapolation.cli fit large-dataset/be_1po --observable Energy
```

Required behavior:

```bash
extrapolate data init
extrapolate data sync
extrapolate data list
extrapolate data show large-dataset/be_1po

extrapolate fit large-dataset/be_1po --observable Energy
extrapolate fit large-dataset/be_1po --all
```

Useful optional outputs:

```bash
--json outputs/result.json
--csv outputs/summary.csv
--plot-dir outputs/figures
--verbose
```

The CLI must call the public Python API rather than reproduce fitting logic.

Keep the existing `tools/data_db.py` working unless intentionally replaced with a documented compatibility wrapper.

### 11. Add reproducible study configuration

Support repeatable multi-dataset runs through a small YAML or TOML configuration.

Example:

```yaml
study: be_states

datasets:
  - large-dataset/be_1po
  - large-dataset/be_3po

observables:
  - Energy
  - MV

fit:
  method: linearized
  n_fit: 6
  compute_uq: true

outputs:
  directory: outputs/be_states
  summary_csv: summary.csv
```

Run with a command such as:

```bash
extrapolate run studies/be_states.yaml
```

Do not build a generic workflow engine.

Study execution must call the same public fitting functions used interactively.

Configuration parsing must validate unknown keys and malformed values.

### 12. Add run manifests

When users explicitly export a result or study, write a small reproducibility manifest.

Include, when available:

- logical dataset name;
- source hash or dataset version metadata;
- observable;
- independent-variable column;
- selected row identifiers or range;
- fitting method;
- model;
- numerical options;
- uncertainty options;
- package version;
- database schema version;
- timestamp;
- output file list.

Do not automatically store manifests in DuckDB.

Do not require manifests for interactive fitting.

---

## Scientific Methodology That Must Remain Stable

The project uses a hybrid linear/nonlinear parameter-scan approach for asymptotic fitting.

The core behavior includes:

1. scan candidate asymptotic baseline values;
2. subtract each candidate baseline from the dependent variable;
3. reject candidates that violate the logarithm domain;
4. logarithmically linearize the remaining relationship;
5. perform linear regression;
6. choose the optimum using the established fit criterion;
7. quantify baseline uncertainty from profile degradation around the optimum.

Do not replace this with an unrelated nonlinear optimizer.

Do not change scan ranges, numerical tolerances, uncertainty thresholds, parameter meanings, default model interpretation, or result units unless a targeted test demonstrates a bug and the reason is documented.

Use vectorized NumPy/SciPy operations where practical.

Handle invalid logarithm candidates deliberately rather than through uncontrolled warnings.

---

## Public API Design Rules

Prefer a small, discoverable API.

Good:

```python
result = fit_dataset("large-dataset/be_1po", "Energy")
```

Avoid requiring users to understand internal constructor details for routine work.

Keep lower-level access available:

```python
solver = VarProLinearized(...)
solver.fit_linearized(...)
```

Use keyword-only arguments for optional fitting controls where practical.

Avoid domain-specific Boolean arguments in new public APIs when a named mode is clearer.

If `use_energy_b` must remain, document it clearly and consider a future named alternative while preserving compatibility.

Do not introduce deeply nested factories, dependency-injection frameworks, registries, or plugin systems.

---

## Compatibility

Preserve current notebooks and scripts where practical.

Use deprecation warnings rather than abrupt removal for commonly used imports or functions.

Compatibility wrappers must be thin and must call the canonical implementation.

Do not maintain two independent code paths indefinitely.

Document changed imports and provide direct replacements.

---

## Testing Requirements

Use temporary directories and temporary DuckDB files.

Do not modify the developer's real database.

Do not require network access.

Add focused tests for packaging, high-level fitting, numerical agreement with the existing solver, metadata defaults, relationship resolution, validation failures, selection behavior, JSON serialization, headless plotting, batch delegation, CLI smoke tests, and study configuration.

Use compact fixtures instead of copying the full production dataset into tests.

---

## Documentation Requirements

Update the README with one clear quick start.

It should include:

```bash
python -m pip install -e .
python tools/data_db.py init
python tools/data_db.py sync
```

Then:

```python
from extrapolation import fit_dataset

result = fit_dataset(
    "large-dataset/be_1po",
    observable="Energy",
)

print(result.summary())
result.plot()
```

Document the role of source files and DuckDB, dataset discovery, single and batch fitting, study runs, exports, lower-level solver access, automatic related-dataset association, and metadata overrides.

Keep examples runnable.

---

## Implementation Order

1. inspect the current package, fitting helpers, plotting code, tests, and notebooks;
2. establish `pyproject.toml` and package imports;
3. add `FitResult`;
4. add dataset bundle/relationship resolution;
5. add validation and explicit row selection;
6. implement canonical `fit_dataset()`;
7. make plotting consume `FitResult`;
8. refactor batch helpers to delegate to `fit_dataset()`;
9. add CLI;
10. add study configuration and manifests;
11. update documentation;
12. run tests and smoke tests;
13. inspect the final diff for unrelated changes.

Keep each logical change reviewable.

Do not perform broad formatting unrelated to this work.

---

## Validation Commands

At minimum, attempt:

```bash
python -m pip install -e .
python -m pytest
python -m compileall src
```

Run a fresh-database smoke test in a temporary location and a representative high-level fit for `large-dataset/be_1po`, observable `Energy`.

Run CLI smoke tests using the final supported syntax.

Use a noninteractive matplotlib backend for automated plot tests.

Do not claim success for commands that were not executed.

---

## Code Quality

Prefer small functions, dataclasses for value objects, public type hints, explicit exceptions, `pathlib.Path`, deterministic behavior, pandas/NumPy interoperability, narrow module responsibilities, one canonical fitting path, and portable serialization.

Avoid hidden global database connections, SQL inside numerical solvers, duplicated fitting implementations, import-time side effects, mutable defaults, silent row dropping, silent guessing, automatic fit persistence, and unnecessary framework abstractions.

---

## Scope Control

Do not redesign the DuckDB schema, redesign the core mathematics, delete source files, rewrite all notebooks, create a GUI or web service, introduce cloud infrastructure or an ORM, build a generic workflow engine, store every fit in DuckDB, rename the whole repository, or perform unrelated cleanup.

Make the smallest coherent implementation that creates a polished, reusable fitting workflow.

---

## Git Safety

Before editing:

```bash
git status --short
git branch --show-current
```

Do not discard uncommitted user changes.

Do not run:

```bash
git reset --hard
git clean -fd
git checkout -- .
```

Do not force-push or amend existing commits unless explicitly requested.

Do not commit generated databases, plots, manifests, caches, or temporary study outputs unless explicitly intended.

---

## Completion Criteria

The task is complete only when a fresh user can install, initialize, synchronize, and then run:

```python
from extrapolation import fit_dataset

result = fit_dataset(
    "large-dataset/be_1po",
    observable="Energy",
)

result.summary()
result.plot()
```

without manually reading CSV files, manually locating related files, modifying `sys.path`, manually wiring DataFrames into the solver, or understanding internal solver constructor details for a routine fit.

The lower-level solver must remain available for expert use.

---

## Final Deliverable

At completion, report:

1. final public Python API;
2. package/import structure;
3. files added, modified, or removed;
4. compatibility behavior and deprecations;
5. dataset relationship-resolution rules;
6. validation and row-selection behavior;
7. CLI commands;
8. study configuration format;
9. result and manifest export formats;
10. tests and smoke tests executed;
11. failures or remaining limitations;
12. `git status --short`;
13. a suggested commit message.

Do not return only a design proposal. Implement the workflow, inspect the resulting code, and run the available validation commands.