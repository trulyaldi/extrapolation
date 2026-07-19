from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from database import DatasetDatabase, DatasetImportError, DatasetValidationError
from functions import load_fit_dataset


def _write_sources(root: Path) -> None:
    observations = root / "large-dataset"
    uncertainties = root / "large-dataset-err"
    observations.mkdir(parents=True)
    uncertainties.mkdir()
    (observations / "demo.csv").write_text(
        "Basis Size,Energy,Observable\n"
        "100,-1.0,2.5\n"
        "200,-1.2,\n",
        encoding="utf-8",
    )
    (uncertainties / "demo_err.csv").write_text(
        "Basis Size,Energy,Observable\n"
        "200,0.01,0.02\n",
        encoding="utf-8",
    )


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "sources"
    _write_sources(root)
    return root


@pytest.fixture
def catalog(tmp_path: Path) -> DatasetDatabase:
    db = DatasetDatabase(tmp_path / "catalog.duckdb")
    db.initialize()
    return db


def test_sync_imports_wide_data_missing_values_and_uncertainties(
    catalog: DatasetDatabase, source_root: Path
):
    report = catalog.sync_sources(source_root)

    assert set(report.imported) == {"large-dataset/demo.csv", "large-dataset-err/demo_err.csv"}
    assert catalog.list_datasets()["dataset_name"].tolist() == [
        "large-dataset-err/demo_err",
        "large-dataset/demo",
    ]
    metadata = catalog.get_dataset_metadata("large-dataset/demo")
    assert metadata["source_path"] == "large-dataset/demo.csv"
    assert metadata["columns"] == ["Basis Size", "Energy", "Observable"]
    assert metadata["independent_column"] == "Basis Size"
    assert catalog.get_dataset_metadata("large-dataset-err/demo_err")["data_role"] == "uncertainty"

    loaded = catalog.load_observations("large-dataset/demo")
    assert loaded.columns.tolist() == ["Basis Size", "Energy", "Observable"]
    assert loaded["Basis Size"].tolist() == [100.0, 200.0]
    assert pd.isna(loaded.loc[1, "Observable"])
    with pytest.raises(DatasetValidationError):
        catalog.load_observations("large-dataset-err/demo_err")

    fit_ready = load_fit_dataset("large-dataset/demo", db_path=catalog.db_path)
    assert fit_ready["Basis Size"].tolist() == [100, 200]


def test_sync_is_idempotent_and_preserves_manual_edits(
    catalog: DatasetDatabase, source_root: Path
):
    catalog.sync_sources(source_root)
    catalog.update_value("large-dataset/demo", 0, "Energy", -9.0)

    unchanged = catalog.sync_sources(source_root)
    assert unchanged.unchanged == (
        "large-dataset-err/demo_err.csv",
        "large-dataset/demo.csv",
    )
    assert catalog.load_dataset("large-dataset/demo").loc[0, "Energy"] == -9.0
    assert len(catalog.list_datasets()) == 2

    (source_root / "large-dataset" / "demo.csv").write_text(
        "Basis Size,Energy,Observable\n100,-3.0,2.5\n200,-1.2,\n",
        encoding="utf-8",
    )
    changed = catalog.sync_sources(source_root)
    assert changed.changed == ("large-dataset/demo.csv",)
    assert catalog.load_dataset("large-dataset/demo").loc[0, "Energy"] == -9.0

    replaced = catalog.sync_sources(source_root, replace=True)
    assert replaced.replaced == ("large-dataset/demo.csv",)
    assert catalog.load_dataset("large-dataset/demo").loc[0, "Energy"] == -3.0
    assert not catalog.get_dataset_metadata("large-dataset/demo")["manual_modified"]


def test_sync_requires_explicit_schema_change_permission(
    catalog: DatasetDatabase, source_root: Path
):
    catalog.sync_sources(source_root)
    (source_root / "large-dataset" / "demo.csv").write_text(
        "Basis Size,Energy,New observable\n100,-3.0,4.0\n",
        encoding="utf-8",
    )

    conflict = catalog.sync_sources(source_root, replace=True)
    assert conflict.schema_conflicts == ("large-dataset/demo.csv",)
    assert catalog.get_dataset_metadata("large-dataset/demo")["columns"] == [
        "Basis Size",
        "Energy",
        "Observable",
    ]

    replaced = catalog.sync_sources(source_root, replace=True, allow_schema_change=True)
    assert replaced.replaced == ("large-dataset/demo.csv",)
    assert catalog.get_dataset_metadata("large-dataset/demo")["columns"] == [
        "Basis Size",
        "Energy",
        "New observable",
    ]


def test_sync_rolls_back_when_a_source_is_malformed(catalog: DatasetDatabase, source_root: Path):
    (source_root / "large-dataset" / "broken.csv").write_text(
        "Basis Size,Energy\n100,not-a-number\n", encoding="utf-8"
    )

    with pytest.raises(DatasetImportError, match="broken.csv"):
        catalog.sync_sources(source_root)

    assert catalog.list_datasets().empty


def test_manual_dataset_editing_replacement_export_and_deletion(
    catalog: DatasetDatabase, tmp_path: Path
):
    data = pd.DataFrame({"Basis Size": [100.0], "Energy": [-1.0], "Value": [None]})
    catalog.add_or_replace_dataset("manual/demo", data)
    catalog.update_value("manual/demo", 0, "Value", 2.0)
    new_row = catalog.insert_row("manual/demo", {"Basis Size": 200, "Energy": -1.2})
    assert new_row == 1
    assert pd.isna(catalog.load_dataset("manual/demo").loc[1, "Value"])

    catalog.delete_row("manual/demo", new_row)
    exported = catalog.export_dataset("manual/demo", tmp_path / "export.csv")
    assert exported.exists()
    assert "Basis Size,Energy,Value" in exported.read_text(encoding="utf-8")

    replacement = pd.DataFrame({"Basis Size": [300.0], "Energy": [-1.5]})
    catalog.add_or_replace_dataset("manual/demo", replacement, replace=True)
    assert catalog.load_dataset("manual/demo").columns.tolist() == ["Basis Size", "Energy"]
    catalog.delete_dataset("manual/demo")
    with pytest.raises(KeyError):
        catalog.load_dataset("manual/demo")


def test_user_values_are_parameterized_and_readonly_sql_is_enforced(catalog: DatasetDatabase):
    catalog.add_or_replace_dataset(
        "manual/safe", pd.DataFrame({"Basis Size": [100.0], "Energy": [-1.0]})
    )
    inspected = catalog.execute_readonly_sql(
        "SELECT row_count FROM datasets WHERE dataset_name = ?", ["manual/safe"]
    )
    assert inspected.loc[0, "row_count"] == 1
    with pytest.raises(DatasetValidationError):
        catalog.update_value("manual/safe'; DROP TABLE datasets; --", 0, "Energy", -1.2)
    with pytest.raises(DatasetValidationError):
        catalog.execute_readonly_sql("DELETE FROM datasets")
    assert catalog.load_dataset("manual/safe").loc[0, "Energy"] == -1.0


def test_fitting_path_has_no_result_persistence_api():
    from extrap import VarProLinearized

    assert "save_to_db" not in inspect.signature(VarProLinearized.fit_linearized).parameters
    assert "save_result" not in inspect.getsource(VarProLinearized.fit_linearized)

    fitter = VarProLinearized(
        pd.DataFrame(
            {
                "basis size": [100, 200, 300, 400, 500],
                "Energy": [-1.0, -1.1, -1.16, -1.20, -1.23],
            }
        ),
        "basis size",
        "Energy",
        use_energy_b=False,
    )
    assert set(fitter.fit_linearized(models=["exponential"], compute_uq=False)) == {
        "exponential"
    }
