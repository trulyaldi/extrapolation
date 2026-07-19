"""Public high-level API for reproducible asymptotic extrapolation."""

from extrapolation.api import fit_all_observables, fit_dataset
from extrapolation.data import DatasetBundle, FitValidationError, load_dataset_bundle
from extrapolation.result import FitResult
from extrapolation.study import StudyResult, run_study

__version__ = "0.1.0"

__all__ = [
    "DatasetBundle",
    "FitResult",
    "FitValidationError",
    "StudyResult",
    "fit_all_observables",
    "fit_dataset",
    "load_dataset_bundle",
    "run_study",
]

