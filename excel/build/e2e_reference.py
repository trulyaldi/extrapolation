"""Reference values for the end-to-end test: extraplus called directly, with the
defaults the Excel functions use (k_cov=1, q=0.84, use_PI=True, robust_scale=True).
Deliberately independent of excel/extrapolation.py."""

import numpy as np
import pandas as pd
from extraplus import extraplus

MODELS = {1: "exponential", 2: "sqrt_exponential", 3: "power_law"}

# Be 1Se energy, first 12 basis sizes of research/data/large-dataset/Be_1Se.csv.
DATA = [
    (100, -14.6661876127332),
    (200, -14.6667402339119),
    (300, -14.6672828224583),
    (400, -14.6673249180953),
    (500, -14.66734057696),
    (600, -14.6673445259951),
    (700, -14.6673478029866),
    (800, -14.6673503395393),
    (900, -14.6673519816691),
    (1000, -14.6673531093866),
    (1100, -14.6673536576132),
    (1200, -14.6673541629258),
]


def reference(x, y, n):
    """Return (C, uncertainty, B) for model n in the original y units."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    solver = extraplus(df=pd.DataFrame({"x": x, "y": y}), x_col="x", y_col="y",
                       err_df=None, inf_df=None, b_init=None, n_fit=None, use_energy_b=False)
    solver.fit_linearized(models=[MODELS[n]], verbose=False, compute_uq=False)
    solver.compute_uncertainty(k_cov=1.0, q=0.84, use_PI=True, robust_scale=True)
    res = solver.results[MODELS[n]]
    return (float(solver.y_min + solver.y_range * res["C"]),
            float(solver.y_range * res["sigma_C"]),
            float(res["B"]))
