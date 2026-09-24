"""
extraplus: extrapolation of basis-set sequences to the complete-basis limit.

The public API is :func:`extrapolate` (one model), :func:`compare` (all three
models) and :class:`Result`.  The research engine, the :class:`extraplus`
class, stays importable for existing code.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from ._result import Result
from .api import compare, extrapolate
from .extrap import extraplus

try:
    __version__ = _version("extraplus")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "unknown"

__all__ = ["extrapolate", "compare", "Result", "extraplus", "__version__"]
