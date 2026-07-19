import pandas as pd

from database.results import find_results
from functions import fit_system_summary


def test_fit_system_summary_saves_results():
    df = pd.DataFrame({
        "basis size": [2, 3, 4, 5],
        "energy": [-1.40, -1.60, -1.70, -1.75],
    })

    fit_system_summary(
        df,
        system_name="PYTEST_FIT",
        save_to_db=True,
    )

    rows = find_results("PYTEST_FIT", "energy")
    assert rows
