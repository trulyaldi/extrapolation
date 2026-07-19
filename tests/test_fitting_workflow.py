from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from database import DatasetDatabase
from extrap import VarProLinearized
from extrapolation import (
    FitResult,
    FitValidationError,
    fit_all_observables,
    fit_dataset,
    run_study,
)
from extrapolation.cli import main as cli_main
from extrapolation.database import DatasetDatabase as QualifiedDatasetDatabase
from extrapolation.fitting import VarProLinearized as QualifiedSolver


@pytest.fixture
def fit_catalog(tmp_path: Path) -> DatasetDatabase:
    db = DatasetDatabase(tmp_path / "fit.duckdb")
    db.initialize()
    x = np.arange(1.0, 9.0)
    data = pd.DataFrame(
        {
            "basis size": x,
            "Energy": -2.0 + 0.8 * np.exp(-3.0 * x / x.max()),
            "Observable": 1.25 + 1.1 * np.exp(-2.2 * x / x.max()),
        }
    )
    db.add_or_replace_dataset("large-dataset/demo", data)
    db.add_or_replace_dataset(
        "large-dataset-err/demo_err",
        pd.DataFrame(
            {"basis size": [20.0], "Energy": [1e-5], "Observable": [2e-5]}
        ),
        data_role="uncertainty",
    )
    db.add_or_replace_dataset(
        "large-dataset-init/demo",
        pd.DataFrame(
            {"basis size": [20.0], "Energy": [-2.0], "Observable": [1.25]}
        ),
    )
    return db


def test_package_qualified_imports_are_public():
    assert QualifiedDatasetDatabase is DatasetDatabase
    assert QualifiedSolver is VarProLinearized
    assert FitResult.__module__ == "extrapolation.result"


def test_relationship_resolution_uses_convention_and_explicit_override(
    fit_catalog: DatasetDatabase, monkeypatch,
):
    bundle = fit_catalog.load_bundle("large-dataset/demo")
    assert bundle.error_dataset == "large-dataset-err/demo_err"
    assert bundle.reference_dataset == "large-dataset-init/demo"
    assert bundle.relationship_sources == {
        "error": "naming convention",
        "reference": "naming convention",
    }
    without_related = fit_catalog.load_bundle(
        "large-dataset/demo", error_dataset=None, reference_dataset=None
    )
    assert without_related.errors is None
    assert without_related.reference is None

    fit_catalog.add_or_replace_dataset(
        "manual/preferred-errors",
        pd.DataFrame(
            {"basis size": [20.0], "Energy": [2e-5], "Observable": [3e-5]}
        ),
        data_role="uncertainty",
    )
    original_metadata = fit_catalog.get_dataset_metadata

    def related_metadata(dataset_name):
        metadata = original_metadata(dataset_name)
        if dataset_name == "large-dataset/demo":
            metadata["related_error_dataset"] = "manual/preferred-errors"
        return metadata

    monkeypatch.setattr(fit_catalog, "get_dataset_metadata", related_metadata)
    metadata_bundle = fit_catalog.load_bundle("large-dataset/demo")
    assert metadata_bundle.error_dataset == "manual/preferred-errors"
    assert metadata_bundle.relationship_sources["error"] == "metadata"
    explicit_bundle = fit_catalog.load_bundle(
        "large-dataset/demo",
        error_dataset="large-dataset-err/demo_err",
    )
    assert explicit_bundle.error_dataset == "large-dataset-err/demo_err"
    assert explicit_bundle.relationship_sources["error"] == "argument"


def test_fit_dataset_matches_existing_solver(fit_catalog: DatasetDatabase):
    frame = fit_catalog.load_dataset("large-dataset/demo")
    low_level = VarProLinearized(
        frame, "basis size", "Energy", use_energy_b=False
    ).fit_linearized(models=["exponential"], compute_uq=False)["exponential"]
    expected = float(frame["Energy"].min()) + float(frame["Energy"].max() - frame["Energy"].min()) * low_level["C"]

    result = fit_dataset(
        "large-dataset/demo",
        "Energy",
        db=fit_catalog,
        model="exponential",
        compute_uq=False,
        use_energy_b=False,
    )
    assert result.baseline == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result.parameters["B"] == pytest.approx(low_level["B"])
    assert len(result.scan_values) == 1250
    assert result.metadata["relationships"]["error_dataset"].endswith("demo_err")


def test_selection_defaults_validation_and_missing_policy(
    fit_catalog: DatasetDatabase,
):
    result = fit_dataset(
        "large-dataset/demo",
        "Energy",
        db=fit_catalog,
        model="exponential",
        n_fit=4,
        basis_min=2,
        compute_uq=False,
        use_energy_b=False,
    )
    assert result.selected_data["basis size"].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert result.metadata["selection"]["row_indices"] == [4, 5, 6, 7]

    invalid = fit_catalog.load_dataset("large-dataset/demo")
    invalid.loc[3, "Energy"] = np.nan
    fit_catalog.add_or_replace_dataset("manual/missing", invalid, replace=False)
    dropped = fit_dataset(
        "manual/missing",
        "Energy",
        db=fit_catalog,
        model="exponential",
        compute_uq=False,
        use_energy_b=False,
    )
    assert dropped.metadata["selection"]["dropped_missing_row_indices"] == [3]
    with pytest.raises(FitValidationError, match="missing or non-finite"):
        fit_dataset(
            "manual/missing",
            "Energy",
            db=fit_catalog,
            model="exponential",
            missing="raise",
            compute_uq=False,
            use_energy_b=False,
        )
    with pytest.raises(FitValidationError, match="observable column is absent"):
        fit_dataset("large-dataset/demo", "Absent", db=fit_catalog)


def test_metadata_defaults_apply_and_explicit_arguments_win(
    fit_catalog: DatasetDatabase, monkeypatch
):
    original_metadata = fit_catalog.get_dataset_metadata

    def configured_metadata(dataset_name):
        metadata = original_metadata(dataset_name)
        metadata["fit_defaults"] = {
            "method": "linearized",
            "n_fit": 3,
            "basis_min": 4,
            "use_energy_b": False,
        }
        return metadata

    monkeypatch.setattr(fit_catalog, "get_dataset_metadata", configured_metadata)
    defaulted = fit_dataset(
        "large-dataset/demo",
        "Energy",
        db=fit_catalog,
        model="exponential",
        compute_uq=False,
    )
    assert defaulted.selected_data["basis size"].tolist() == [6.0, 7.0, 8.0]
    explicit = fit_dataset(
        "large-dataset/demo",
        "Energy",
        db=fit_catalog,
        model="exponential",
        n_fit=4,
        basis_min=2,
        use_energy_b=False,
        compute_uq=False,
    )
    assert explicit.selected_data["basis size"].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_result_serialization_and_headless_plots(
    fit_catalog: DatasetDatabase, tmp_path: Path
):
    result = fit_dataset(
        "large-dataset/demo",
        "Energy",
        db=fit_catalog,
        model="exponential",
        compute_uq=False,
        use_energy_b=False,
    )
    payload = json.loads(result.to_json())
    assert payload["dataset_name"] == "large-dataset/demo"
    destination = result.to_json(tmp_path / "result.json")
    assert destination.exists()
    assert (tmp_path / "result.manifest.json").exists()
    assert "baseline=" in result.summary()

    for plotter, name in (
        (result.plot, "fit.png"),
        (result.plot_log, "log.png"),
        (result.plot_profile, "profile.png"),
    ):
        figure, axes = plotter(output_path=tmp_path / name)
        assert figure.axes[0] is axes
        assert (tmp_path / name).exists()
        plt.close(figure)


def test_batch_delegates_to_canonical_function(fit_catalog, monkeypatch):
    import extrapolation.api as api

    calls = []
    real_fit = api.fit_dataset

    def recording_fit(dataset, observable, **kwargs):
        calls.append((dataset, observable))
        return real_fit(dataset, observable, **kwargs)

    monkeypatch.setattr(api, "fit_dataset", recording_fit)
    results = fit_all_observables(
        "large-dataset/demo",
        observables=["Energy"],
        db=fit_catalog,
        model="exponential",
        compute_uq=False,
        use_energy_b=False,
    )
    assert calls == [("large-dataset/demo", "Energy")]
    assert set(results) == {"Energy"}


def test_cli_and_toml_study(fit_catalog: DatasetDatabase, tmp_path: Path, capsys):
    exit_code = cli_main(
        [
            "--db",
            str(fit_catalog.db_path),
            "fit",
            "large-dataset/demo",
            "--observable",
            "Energy",
            "--model",
            "exponential",
            "--no-uq",
            "--no-use-energy-b",
            "--json",
            str(tmp_path / "cli-result.json"),
        ]
    )
    assert exit_code == 0
    assert "baseline=" in capsys.readouterr().out
    assert (tmp_path / "cli-result.json").exists()
    assert (tmp_path / "cli-result.manifest.json").exists()

    all_exit_code = cli_main(
        [
            "--db",
            str(fit_catalog.db_path),
            "fit",
            "large-dataset/demo",
            "--all",
            "--model",
            "exponential",
            "--no-uq",
            "--no-use-energy-b",
        ]
    )
    assert all_exit_code == 0
    assert "Observable" in capsys.readouterr().out

    study_path = tmp_path / "study.toml"
    study_path.write_text(
        'study = "demo"\n'
        'datasets = ["large-dataset/demo"]\n'
        'observables = ["Energy"]\n\n'
        "[fit]\n"
        'model = "exponential"\n'
        "compute_uq = false\n"
        "use_energy_b = false\n\n"
        "[outputs]\n"
        'directory = "study-output"\n'
        'summary_csv = "summary.csv"\n',
        encoding="utf-8",
    )
    study = run_study(study_path, db=fit_catalog)
    assert len(study.results) == 1
    assert study.summary_path and study.summary_path.exists()
    assert study.manifest_path.exists()
    manifest = json.loads(study.manifest_path.read_text(encoding="utf-8"))
    assert manifest["study"] == "demo"

    bad_study = tmp_path / "bad.toml"
    bad_study.write_text(
        'study="bad"\ndatasets=["large-dataset/demo"]\n'
        'observables=["Energy"]\nunknown=true\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level"):
        run_study(bad_study, db=fit_catalog)

    bad_type = tmp_path / "bad-type.toml"
    bad_type.write_text(
        'study="bad"\ndatasets=["large-dataset/demo"]\n'
        'observables=["Energy"]\n[outputs]\nplots="yes"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outputs.plots must be true or false"):
        run_study(bad_type, db=fit_catalog)
