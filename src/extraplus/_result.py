"""The result of one extrapolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ._validation import targets
from .extrap import extraplus

# Plain-text forms of the fitted models (see extraplus._compute_basis).
_FORMULAS = {
    "exponential": "y = A * exp(-B * x/x_max) + C",
    "sqrt_exponential": "y = A * exp(-B * sqrt(x/x_max)) + C",
    "power_law": "y = A * (x/x_max)^(-B) + C",
}

_PUBLIC_FIELDS = (
    "value", "uncertainty", "decay_rate", "amplitude", "r2",
    "model", "n_points", "n_fit_used", "x_max", "b_fixed",
)


@dataclass(frozen=True)
class Result:
    """
    The extrapolation of a sequence with one model, as returned by
    :func:`extraplus.extrapolate`.

    The model is ``y = A * phi(x / x_max) + C`` with ``phi(t) = exp(-B t)``
    (exponential), ``exp(-B sqrt(t))`` (sqrt_exponential) or ``t**(-B)``
    (power_law).  ``value`` is the limit C for x -> infinity, and
    :meth:`predict` gives the fitted y at a finite x.

    Attributes
    ----------
    value : float
        The extrapolated value C, in the units of y.
    uncertainty : float
        The uncertainty of C, in the units of y.  It is ``inf`` when the fit
        does not constrain C.
    decay_rate : float
        B, the fitted decay rate (or ``b_fixed`` when B was fixed), for the
        normalized variable x / x_max.
    amplitude : float
        A, in the units of y.  Positive for a decreasing sequence, negative
        for an increasing one.
    r2 : float
        R^2 of the fit in linearized (log) space.
    model : str
        ``"exponential"``, ``"sqrt_exponential"`` or ``"power_law"``.
    n_points : int
        Number of usable (x, y) pairs in the input.
    n_fit_used : int
        Number of those pairs, from the first, that were fitted.
    x_max : float
        The largest x of all usable pairs (the scale of x / x_max).
    b_fixed : float or None
        The fixed B, or None when B was fitted.

    Examples
    --------
    >>> from extraplus import extrapolate
    >>> result = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03])
    >>> result.model, result.n_points, result.b_fixed
    ('exponential', 5, None)
    >>> sorted(result.to_dict())[:3]
    ['amplitude', 'b_fixed', 'decay_rate']
    >>> result.value < result.predict(1000) < result.predict(600)   # y beyond the data
    True
    """

    value: float
    uncertainty: float
    decay_rate: float
    amplitude: float
    r2: float
    model: str
    n_points: int
    n_fit_used: int
    x_max: float
    b_fixed: Optional[float]
    # The cleaned input, kept to redraw the plots.
    _x: tuple[float, ...] = field(repr=False)
    _y: tuple[float, ...] = field(repr=False)
    # The fit in the engine's scaled units, y_scaled = (y - y_min) / y_range, for predict().
    _C_scaled: float = field(repr=False)
    _A_scaled: float = field(repr=False)
    _y_min: float = field(repr=False)
    _y_range: float = field(repr=False)

    def predict(self, x: ArrayLike) -> Union[float, np.ndarray]:
        """
        The fitted curve ``A * phi(x / x_max) + C`` at the basis size x, in the
        units of y.

        It is meant for x beyond the data, the value a larger basis would give.
        Inside the data range it returns the fitted value there, not the data.
        For x -> infinity it tends to ``value``.  This is the Excel functions
        FINITE_EXTRAP1 to FINITE_EXTRAP3.

        Parameters
        ----------
        x : float or array-like
            The basis size, or several of them (a list, tuple, NumPy array or
            pandas Series).  Each must be a finite number greater than zero.

        Returns
        -------
        float or numpy.ndarray
            A float for a scalar x, otherwise an array of the same shape as x.

        Raises
        ------
        ValueError
            If x (the target) is missing, not a number, not finite, or not
            greater than zero, for example "target must be greater than zero."

        Examples
        --------
        >>> from extraplus import extrapolate
        >>> result = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03])
        >>> y_600 = result.predict(600)
        >>> result.value < y_600 < 1.03
        True
        >>> result.predict([600, 700, 800]).shape
        (3,)
        """
        t = targets(x)
        # The engine's own basis function and unscaling, as in fit_linearized() and plot().
        engine = object.__new__(extraplus)
        engine.model_type = self.model
        phi = engine._compute_basis(self.decay_rate, t.reshape(-1) / self.x_max)[:, 1]
        y = self._y_min + self._y_range * (self._C_scaled + self._A_scaled * phi)
        return float(y[0]) if t.ndim == 0 else y.reshape(t.shape)

    def summary(self) -> str:
        """
        A short, readable description of the result.

        Returns
        -------
        str
            Several lines of text; printing it is up to the caller.

        Examples
        --------
        >>> from extraplus import extrapolate
        >>> result = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03])
        >>> print(result.summary().splitlines()[0])
        Extrapolation with the exponential model, y = A * exp(-B * x/x_max) + C
        """
        how_b = f"fixed at {self.b_fixed:.6g}" if self.b_fixed is not None else "fitted"
        return "\n".join([
            f"Extrapolation with the {self.model} model, {_FORMULAS[self.model]}",
            f"  value (C)        {self.value:.15g}",
            f"  uncertainty      {self.uncertainty:.3g}",
            f"  decay_rate (B)   {self.decay_rate:.6g} ({how_b})",
            f"  amplitude (A)    {self.amplitude:.6g}",
            f"  r2 (linearized)  {self.r2:.6f}",
            f"  points           {self.n_points} ({self.n_fit_used} fitted), x_max = {self.x_max:g}",
        ])

    def to_dict(self) -> dict[str, Any]:
        """
        The result as a dictionary of plain Python values.

        Returns
        -------
        dict
            The keys ``value``, ``uncertainty``, ``decay_rate``, ``amplitude``,
            ``r2``, ``model``, ``n_points``, ``n_fit_used``, ``x_max`` and
            ``b_fixed``.

        Examples
        --------
        >>> from extraplus import extrapolate
        >>> result = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03])
        >>> result.to_dict()["model"]
        'exponential'
        """
        return {name: getattr(self, name) for name in _PUBLIC_FIELDS}

    def plot(self) -> Figure:
        """
        Plot the data, the fitted curve out to 1.5 x_max, and C +- uncertainty.

        The whole x range is shown; points beyond ``n_fit`` are drawn as open
        markers.  The y axis is linear within the uncertainty of C and
        logarithmic in the distance from C beyond that, so the band stays
        visible next to the full data range.  This is the plot of the Excel
        functions EXTRAP1_PLOT to EXTRAP3_PLOT.

        Returns
        -------
        matplotlib.figure.Figure
            An open pyplot figure: show it, save it with ``fig.savefig(...)``,
            and close it with ``matplotlib.pyplot.close(fig)`` when done.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from extraplus import extrapolate
        >>> fig = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03]).plot()
        >>> fig.axes[0].get_xlabel()
        'Basis Size'
        >>> plt.close(fig)
        """
        from . import _plotting

        solver = self._solver()
        return _plotting.fit_figure(solver, self.model, np.asarray(self._x), np.asarray(self._y))

    def plot_linearized(self) -> Figure:
        """
        Plot the fit in linearized space, ln|y - C| against the model's
        transformed x, with its 95% confidence band.

        This is the plot of the Excel functions EXTRAP1_PLOT_LOG to
        EXTRAP3_PLOT_LOG.

        Returns
        -------
        matplotlib.figure.Figure
            An open pyplot figure: show it, save it with ``fig.savefig(...)``,
            and close it with ``matplotlib.pyplot.close(fig)`` when done.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from extraplus import extrapolate
        >>> fig = extrapolate([100, 200, 300, 400, 500], [1.5, 1.2, 1.1, 1.05, 1.03]).plot_linearized()
        >>> fig.axes[0].get_title().splitlines()[1]
        'linearized: $\\\\ln(y-C)$ vs $x/x_{max}$'
        >>> plt.close(fig)
        """
        from . import _plotting

        return _plotting.linearized_figure(self._solver(), self.model)

    def _solver(self):
        """Refit the same data (deterministic) to get an extraplus solver for plotting."""
        from .api import _solve

        n_fit = self.n_fit_used if self.n_fit_used < self.n_points else None
        return _solve(np.asarray(self._x), np.asarray(self._y), self.model, self.b_fixed, n_fit)
