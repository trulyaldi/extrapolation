"""
Extrapolate a sequence y(x) to x -> infinity: the public functions.

The fitting itself is done by the research engine, the ``extraplus`` class
(extrap.py); this module validates the input, runs one model, and returns the
result in the units of y.  The Excel functions call these same functions.
"""

from __future__ import annotations

import math
import warnings
from contextlib import redirect_stdout
from io import StringIO
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._result import Result
from ._validation import MODELS, clean_xy, model_name, optional_b_fixed, optional_n_fit
from .extrap import extraplus

ModelName = Literal["exponential", "sqrt_exponential", "power_law"]
Model = Union[ModelName, Literal[1, 2, 3]]

_X_COL = "__x__"
_Y_COL = "__y__"


def extrapolate(
    x: ArrayLike,
    y: ArrayLike,
    model: Model = "exponential",
    b_fixed: Optional[float] = None,
    n_fit: Optional[int] = None,
) -> Result:
    """
    Extrapolate y(x) to x -> infinity with one model.

    Fits ``y = A * phi(x / x_max) + C`` and returns the limit C with its
    uncertainty.  ``phi(t)`` is ``exp(-B t)`` for "exponential" (model 1),
    ``exp(-B sqrt(t))`` for "sqrt_exponential" (2) and ``t**(-B)`` for
    "power_law" (3), the same numbering as the Excel functions EXTRAP1-3.

    Parameters
    ----------
    x, y : array-like
        Basis sizes and the values computed with them, in the same order: a
        list, tuple, NumPy array or pandas Series.  Pairs where x or y is
        missing (None, ""), text or not finite are skipped.  At least three
        usable pairs are needed.
    model : {"exponential", "sqrt_exponential", "power_law", 1, 2, 3}, optional
        The model to fit.  Default "exponential".
    b_fixed : float, optional
        Fix B at this positive value instead of fitting it.
    n_fit : int, optional
        Fit only the first ``n_fit`` usable pairs (at least 3; truncated to a
        whole number; larger values mean all pairs).  The rest are still shown
        by ``Result.plot()``.

    Returns
    -------
    Result
        C as ``value``, its ``uncertainty``, B as ``decay_rate``, A as
        ``amplitude``, and how the fit was made.

    Raises
    ------
    ValueError
        If the input cannot be fitted, with the message the Excel functions
        show after "Error: ", for example "At least three paired numeric x/y
        observations are required."

    Warns
    -----
    RuntimeWarning
        If the uncertainty is not finite (the fit does not constrain C).

    Examples
    --------
    >>> from extraplus import extrapolate
    >>> x = [100, 200, 300, 400, 500, 600]
    >>> y = [-14.66619, -14.66674, -14.66728, -14.66732, -14.66734, -14.66734]
    >>> result = extrapolate(x, y, model="power_law")
    >>> result.model == extrapolate(x, y, model=3).model
    True
    >>> result.value < min(y)
    True
    """
    model_type = model_name(model)
    x_clean, y_clean, fixed_b, n_fit_value = _prepare(x, y, b_fixed, n_fit)
    return _fit(x_clean, y_clean, model_type, fixed_b, n_fit_value)


def compare(
    x: ArrayLike,
    y: ArrayLike,
    b_fixed: Optional[float] = None,
    n_fit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extrapolate y(x) with all three models.

    Parameters
    ----------
    x, y : array-like
        As for :func:`extrapolate`.
    b_fixed : float, optional
        As for :func:`extrapolate`; applied to every model.
    n_fit : int, optional
        As for :func:`extrapolate`.

    Returns
    -------
    pandas.DataFrame
        One row per model, indexed by model name ("exponential",
        "sqrt_exponential", "power_law"), with the columns ``value``,
        ``uncertainty``, ``decay_rate``, ``amplitude``, ``r2``, ``n_points``,
        ``n_fit_used``, ``x_max`` and ``b_fixed`` (see :class:`Result`).

    Raises
    ------
    ValueError
        If the input cannot be fitted, as for :func:`extrapolate`.

    Warns
    -----
    RuntimeWarning
        For each model whose uncertainty is not finite.

    Examples
    --------
    >>> from extraplus import compare
    >>> x = [100, 200, 300, 400, 500, 600]
    >>> y = [-14.66619, -14.66674, -14.66728, -14.66732, -14.66734, -14.66734]
    >>> table = compare(x, y)
    >>> list(table.index)
    ['exponential', 'sqrt_exponential', 'power_law']
    >>> list(table.columns[:3])
    ['value', 'uncertainty', 'decay_rate']
    """
    x_clean, y_clean, fixed_b, n_fit_value = _prepare(x, y, b_fixed, n_fit)
    rows = [_fit(x_clean, y_clean, model, fixed_b, n_fit_value).to_dict() for model in MODELS]
    return pd.DataFrame(rows).set_index("model")


def _prepare(x, y, b_fixed, n_fit):
    """Validate the input once: cleaned x, y, and the checked b_fixed and n_fit."""
    x_clean, y_clean = clean_xy(x, y)
    return x_clean, y_clean, optional_b_fixed(b_fixed), optional_n_fit(n_fit, len(x_clean))


def _solve(x: np.ndarray, y: np.ndarray, model: str, b_fixed, n_fit) -> extraplus:
    """
    Run ``model`` through the extraplus engine on validated input and return the
    solver (its ``results[model]`` hold the scaled fit and uncertainty).

    ``use_energy_b=False`` is deliberate: B from another property is passed
    explicitly as ``b_fixed``.  Everything extraplus prints is discarded.
    """
    df = pd.DataFrame({_X_COL: x, _Y_COL: y})

    with redirect_stdout(StringIO()):
        solver = extraplus(
            df=df,
            x_col=_X_COL,
            y_col=_Y_COL,
            err_df=None,
            inf_df=None,
            b_init=b_fixed,
            n_fit=n_fit,
            use_energy_b=False,
        )

        # Fit first without UQ, then compute the uncertainty in its own call.
        results = solver.fit_linearized(
            models=[model],
            verbose=False,
            compute_uq=False,
        )

        if model not in results:
            raise ValueError(f"extraplus did not return a result for '{model}'.")

        solver.compute_uncertainty()

    return solver


def _fit(x: np.ndarray, y: np.ndarray, model: str, b_fixed, n_fit) -> Result:
    """Fit validated input and convert the scaled extraplus results to the units of y."""
    solver = _solve(x, y, model, b_fixed, n_fit)
    result = solver.results[model]

    C_scaled = float(result["C"])
    A_scaled = float(result["A"])

    if "sigma_C" in result:
        sigma_C_scaled = float(result["sigma_C"])
    elif "sigma_mc" in result:
        # Compatibility with older extraplus versions.
        sigma_C_scaled = float(result["sigma_mc"])
    else:
        raise ValueError("extraplus did not produce sigma_C uncertainty.")

    fitted = Result(
        value=float(solver.y_min + solver.y_range * C_scaled),
        uncertainty=float(solver.y_range * sigma_C_scaled),
        decay_rate=float(result["B"]),
        amplitude=float(solver.y_range * A_scaled),
        r2=float(result["r2_linearized"]),
        model=model,
        n_points=int(len(x)),
        n_fit_used=int(len(solver.raw_x)),
        x_max=float(solver.x_max),
        b_fixed=b_fixed,
        _x=tuple(float(value) for value in x),
        _y=tuple(float(value) for value in y),
        _C_scaled=C_scaled,
        _A_scaled=A_scaled,
        _y_min=float(solver.y_min),
        _y_range=float(solver.y_range),
    )

    if not math.isfinite(fitted.uncertainty):
        warnings.warn(
            f"The uncertainty of the {model} extrapolation is not finite: the fit "
            "does not constrain C, so treat the value with caution.",
            RuntimeWarning,
            stacklevel=3,
        )
    return fitted
