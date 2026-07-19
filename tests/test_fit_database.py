import pandas as pd

from database.results import find_results
from functions import fit_system_summary


def test_fit_system_summary_saves_scan_and_uq_metadata():
    df = pd.DataFrame({
        "basis size": [2, 3, 4, 5],
        "energy": [-1.40, -1.60, -1.70, -1.75],
    })

    fit_system_summary(
        df,
        system_name="PYTEST_METADATA",
        save_to_db=True,
    )

    rows = find_results("PYTEST_METADATA", "energy")
    assert rows

    metadata = rows[0]["metadata"]
    assert metadata["scan_config"]["n_coarse"] == 1250
    assert metadata["scan_config"]["polish"] is True
    assert metadata["uq_config"]["tail_frac"] == 0.5
    assert metadata["uq_config"]["robust_scale"] is True