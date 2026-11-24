"""
Excel functions (xlwings UDFs) for extraplus.

Each function converts its Excel ranges, calls the extraplus Python API
(``extraplus.extrapolate()``, ``Result.predict()``, ``Result.plot()``,
``Result.plot_linearized()``) and formats the result for the cell, so Excel and
Python give the same results by construction.  This module contains no fitting, validation or plotting
logic of its own.

Excel functions:

    EXTRAP1 / EXTRAP2 / EXTRAP3
        Return the extrapolated infinite-limit value C.

    FINITE_EXTRAP1 / FINITE_EXTRAP2 / FINITE_EXTRAP3
        Return the fitted curve's value at a finite target basis size.

    UNCERTAINTY1 / UNCERTAINTY2 / UNCERTAINTY3
        Return the uncertainty of C in the original y units.

    DECAY_RATE1 / DECAY_RATE2 / DECAY_RATE3
        Return the fitted/fixed B parameter.

    EXTRAP1_V / EXTRAP2_V / EXTRAP3_V
        Spill [C, uncertainty, B] vertically.

    EXTRAP1_H / EXTRAP2_H / EXTRAP3_H
        Spill [C, uncertainty, B] horizontally.

    EXTRAP_ALL_V
        All three models in one 3 x 3 array: column n is EXTRAPn_V.

    EXTRAP1_PLOT / EXTRAP2_PLOT / EXTRAP3_PLOT
        Insert the fit plot (Result.plot()) below the cell.

    EXTRAP1_PLOT_LOG / EXTRAP2_PLOT_LOG / EXTRAP3_PLOT_LOG
        Insert the linearized-space plot (Result.plot_linearized()) below the cell.

    EXTRAPLUS_VERSION
        Return the installed extraplus package version.

    EXTRAPLUS_CACHE_STATS
        Return the state of the result cache (see "Result cache" below).

Model mapping:

    1 -> exponential
    2 -> sqrt_exponential
    3 -> power_law

Invalid input does not raise into Excel: the non-plot functions return the
API's ValueError message after "Error: " (for example "Error: At least three
paired numeric x/y observations are required.").

Installation: the ``extraplus`` package must be importable. The Windows
installer (excel/install.bat) installs the package, this module and the
``extraplus.xlam`` add-in into a per-user folder; nothing has to sit next to
the workbook. See excel/README.md.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import struct
import sys
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import fields
from functools import wraps
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Import the installed ``extraplus`` package.
# ---------------------------------------------------------------------------

try:
    import extraplus
    from extraplus import Result, extrapolate
except ImportError as exc:  # pragma: no cover - installation/configuration error
    raise ImportError(
        "Could not import 'extraplus'. Install the extraplus package into this "
        "Python environment (on Windows, run excel/install.bat)."
    ) from exc


# xlwings is optional for normal Python testing, but required inside Excel.
try:
    import xlwings as xw
except ImportError:  # pragma: no cover
    class _XlwingsStub:
        def func(self, function=None, **_kwargs):
            if function is None:
                return lambda fn: fn
            return function

        def arg(self, *_args, **_kwargs):
            return lambda fn: fn

        def ret(self, *_args, **_kwargs):
            return lambda fn: fn

    xw = _XlwingsStub()  # type: ignore[assignment]


_EXPONENTIAL = "exponential"
_SQRT_EXPONENTIAL = "sqrt_exponential"
_POWER_LAW = "power_law"


# ---------------------------------------------------------------------------
# Calling the API from a cell
# ---------------------------------------------------------------------------


def _cells(values: Any) -> tuple:
    """The values of an Excel range (or a scalar) in order, as a hashable tuple."""
    return tuple(np.asarray(values, dtype=object).reshape(-1))


def _extrapolate_quietly(x, y, model_type, b_fixed, n_fit) -> Result:
    """extrapolate() without its RuntimeWarning, which has nowhere to go in Excel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return extrapolate(x, y, model=model_type, b_fixed=b_fixed, n_fit=n_fit)


def _run_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    b_fixed: Any = None,
    n_fit: Any = None,
) -> Result:
    """Shared implementation used by every numeric xlwings UDF below."""
    return _CACHE.extrapolate(_cells(x_range), _cells(y_range), model_type, b_fixed, n_fit)


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------
#
# Every numeric function is a function of one fit, extrapolate(x, y, model,
# b_fixed, n_fit), so the fit's Result is cached and each function formats it
# as before.  Two levels: an in-memory LRU in the UDF server process, and an
# SQLite file shared by every Excel instance of this install.
#
# Key: SHA-256 of the schema tag, the code fingerprint (extraplus version and
# source, numpy/scipy/pandas and Python versions), the model, and the exact
# cell values: each value is tagged with its type and floats are their 8
# IEEE-754 bytes, so 0.0 and -0.0, every NaN payload, 1 and 1.0, True and 1,
# "1" and 1 all give different keys.  Values of any other type (dates, ranges
# passed as b_fixed, ...) are not cached.  Only successful fits are stored: a
# failure (ValueError or anything else) is recomputed every time.
#
# Switches (environment of the Excel process, read when the module loads):
#   EXTRAPLUS_CACHE      "off" (also 0/false/no): no cache; "memory": no file;
#                        anything else (default): memory and file
#   EXTRAPLUS_CACHE_DIR  folder for the file (default %EXTRAPLUS_HOME%\resultcache;
#                        without either variable the cache is memory only)
#
# The file never breaks a function: when it is locked by another Excel for
# longer than _DISK_BUSY_TIMEOUT_S the call computes without it, a corrupt
# file is renamed to *.corrupt-<time> and replaced, and after
# _DISK_ERROR_LIMIT unexplained errors the process stops using it.

_CACHE_SCHEMA = b"extraplus-udf-cache/1"
_MEMORY_ENTRIES = 20_000          # Results kept in memory (LRU)
_DISK_ENTRIES = 100_000           # rows kept in the file (oldest written are deleted first)
_DISK_EVICT_EVERY = 1_000         # check the row count after this many writes
_DISK_BUSY_TIMEOUT_S = 0.25
_DISK_ERROR_LIMIT = 3
_DISK_FILE = "results.sqlite3"
_RESULT_FIELDS = tuple(f.name for f in fields(Result))


class _Uncacheable(Exception):
    """A value the cache key or the stored Result cannot represent exactly."""


def _encode(value: Any, out: bytearray) -> None:
    """Append the exact, type-tagged bytes of one cell value (or Result field)."""
    if value is None:
        out += b"n"
    elif isinstance(value, (bool, np.bool_)):
        out += b"b1" if value else b"b0"
    elif isinstance(value, (float, np.floating)):
        out += b"f" + struct.pack("<d", float(value))
    elif isinstance(value, (int, np.integer)):
        text = str(int(value)).encode()
        out += b"i" + struct.pack("<I", len(text)) + text
    elif isinstance(value, str):
        text = value.encode("utf-8", "surrogatepass")
        out += b"s" + struct.pack("<I", len(text)) + text
    elif isinstance(value, tuple):
        out += b"t" + struct.pack("<I", len(value))
        for item in value:
            _encode(item, out)
    else:
        raise _Uncacheable(type(value).__name__)


def _decode(data: bytes, pos: int) -> tuple[Any, int]:
    tag = data[pos:pos + 1]
    pos += 1
    if tag == b"n":
        return None, pos
    if tag == b"b":
        return data[pos:pos + 1] == b"1", pos + 1
    if tag == b"f":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if tag in (b"i", b"s"):
        (size,) = struct.unpack_from("<I", data, pos)
        text = data[pos + 4:pos + 4 + size]
        if len(text) != size:
            raise _Uncacheable("truncated")
        pos += 4 + size
        return (int(text) if tag == b"i" else text.decode("utf-8", "surrogatepass")), pos
    if tag == b"t":
        (size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        items = []
        for _ in range(size):
            item, pos = _decode(data, pos)
            items.append(item)
        return tuple(items), pos
    raise _Uncacheable(f"tag {tag!r}")


def _encode_result(result: Result) -> bytes:
    out = bytearray()
    _encode(tuple(getattr(result, name) for name in _RESULT_FIELDS), out)
    return bytes(out) + hashlib.sha256(out).digest()[:8]


def _decode_result(blob: bytes) -> Result:
    payload, checksum = blob[:-8], blob[-8:]
    if hashlib.sha256(payload).digest()[:8] != checksum:
        raise _Uncacheable("checksum")
    values, end = _decode(payload, 0)
    if end != len(payload) or not isinstance(values, tuple) or len(values) != len(_RESULT_FIELDS):
        raise _Uncacheable("layout")
    return Result(**dict(zip(_RESULT_FIELDS, values)))


def _code_fingerprint() -> bytes:
    """What the cached Results depend on besides the input: the fitting code and its libraries."""
    h = hashlib.sha256()
    h.update(f"extraplus {extraplus.__version__}\n".encode())
    package = Path(extraplus.__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        h.update(path.name.encode() + b"\n" + path.read_bytes())
    for name in ("numpy", "scipy", "pandas"):
        try:
            h.update(f"{name} {_package_version(name)}\n".encode())
        except PackageNotFoundError:
            h.update(f"{name} ?\n".encode())
    h.update(sys.version.encode())
    return h.digest()


class _DiskCache:
    """The SQLite level.  Every method returns quietly instead of raising."""

    def __init__(self, path: Path, fingerprint: bytes):
        self.path = path
        self.fingerprint = fingerprint[:8]
        self.state = "not opened"
        self.lock = threading.Lock()
        self._con: Optional[sqlite3.Connection] = None
        self._errors = 0
        self._writes = 0

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.path), timeout=_DISK_BUSY_TIMEOUT_S,
                              isolation_level=None, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("CREATE TABLE IF NOT EXISTS results ("
                        "key BLOB PRIMARY KEY, fingerprint BLOB NOT NULL, value BLOB NOT NULL)")
        except BaseException:
            con.close()
            raise
        try:   # rows from other versions can never match again; skipped if another Excel is busy
            con.execute("DELETE FROM results WHERE fingerprint <> ?", (self.fingerprint,))
        except sqlite3.OperationalError as exc:
            if not self._is_busy(exc):
                con.close()
                raise
        return con

    def _open(self) -> Optional[sqlite3.Connection]:
        if self._con is not None or self.state.startswith(("disabled", "failed")):
            return self._con
        try:
            self._con, self.state = self._connect(), "open"
        except Exception as exc:
            self._failed(exc)
        return self._con

    @staticmethod
    def _is_corrupt(exc: Exception) -> bool:
        return isinstance(exc, sqlite3.DatabaseError) and (
            getattr(exc, "sqlite_errorname", "") in ("SQLITE_CORRUPT", "SQLITE_NOTADB")
            or "not a database" in str(exc) or "malformed" in str(exc))

    @staticmethod
    def _is_busy(exc: Exception) -> bool:
        return isinstance(exc, sqlite3.OperationalError) and (
            getattr(exc, "sqlite_errorname", "") in ("SQLITE_BUSY", "SQLITE_LOCKED")
            or "locked" in str(exc) or "busy" in str(exc))

    def _quarantine(self) -> bool:
        """Move a corrupt file aside so that the next open starts a new one."""
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            self.path.replace(self.path.with_name(f"{self.path.name}.corrupt-{stamp}"))
            for suffix in ("-wal", "-shm"):
                side = self.path.with_name(self.path.name + suffix)
                if side.exists():
                    side.unlink()
            return True
        except OSError:   # e.g. still open in another Excel
            return False

    def _failed(self, exc: Exception) -> None:
        if self._is_busy(exc):
            return   # another Excel holds the file: do without it this time
        if self._is_corrupt(exc) and self._quarantine():
            self.state = "not opened"   # the next call starts a new file
            return
        self.close()
        self._errors += 1
        self.state = (f"disabled after {self._errors} errors: {exc}" if self._errors >= _DISK_ERROR_LIMIT
                      else f"error: {exc}")

    def get(self, key: bytes) -> Optional[Result]:
        with self.lock:
            con = self._open()
            if con is None:
                return None
            try:
                row = con.execute("SELECT value FROM results WHERE key = ?", (key,)).fetchone()
            except Exception as exc:
                self._failed(exc)
                return None
            if row is None:
                return None
            try:
                return _decode_result(bytes(row[0]))
            except Exception:   # a damaged row: recompute, and drop it
                try:
                    con.execute("DELETE FROM results WHERE key = ?", (key,))
                except Exception as exc:
                    self._failed(exc)
                return None

    def put(self, key: bytes, result: Result) -> None:
        with self.lock:
            con = self._open()
            if con is None:
                return
            try:
                con.execute("INSERT OR IGNORE INTO results (key, fingerprint, value) VALUES (?, ?, ?)",
                            (key, self.fingerprint, _encode_result(result)))
                self._writes += 1
                if self._writes % _DISK_EVICT_EVERY == 0:
                    self._evict(con)
            except Exception as exc:
                self._failed(exc)

    def _evict(self, con: sqlite3.Connection) -> None:
        (count,) = con.execute("SELECT count(*) FROM results").fetchone()
        if count > _DISK_ENTRIES:
            excess = count - int(0.9 * _DISK_ENTRIES)
            con.execute("DELETE FROM results WHERE rowid IN "
                        "(SELECT rowid FROM results ORDER BY rowid LIMIT ?)", (excess,))

    def count(self) -> Optional[int]:
        with self.lock:
            con = self._open()
            if con is None:
                return None
            try:
                return con.execute("SELECT count(*) FROM results").fetchone()[0]
            except Exception as exc:
                self._failed(exc)
                return None

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
        self._con = None


class _ResultCache:
    def __init__(self, environ=os.environ):
        mode = environ.get("EXTRAPLUS_CACHE", "on").strip().lower()
        self.mode = ("off" if mode in ("off", "0", "false", "no")
                     else "memory" if mode == "memory" else "on")
        self._lock = threading.Lock()
        self._memory: OrderedDict = OrderedDict()
        self.hits_memory = self.hits_disk = self.misses = self.uncacheable = 0
        self._prefix: Optional[bytes] = None
        self._fingerprint: Optional[bytes] = None
        self.disk: Optional[_DiskCache] = None
        folder = environ.get("EXTRAPLUS_CACHE_DIR") or (
            str(Path(environ["EXTRAPLUS_HOME"]) / "resultcache") if environ.get("EXTRAPLUS_HOME") else "")
        self._disk_path = Path(folder) / _DISK_FILE if (folder and self.mode == "on") else None

    def _key(self, x_cells, y_cells, model_type, b_fixed, n_fit) -> bytes:
        if self._prefix is None:
            self._fingerprint = _code_fingerprint()
            self._prefix = _CACHE_SCHEMA + self._fingerprint
            if self._disk_path is not None:
                self.disk = _DiskCache(self._disk_path, self._fingerprint)
        out = bytearray(self._prefix)
        _encode(model_type, out)
        for cells in (x_cells, y_cells):
            _encode(tuple(cells), out)
        _encode(b_fixed, out)
        _encode(n_fit, out)
        return hashlib.sha256(out).digest()

    def extrapolate(self, x_cells: tuple, y_cells: tuple, model_type: str,
                    b_fixed: Any, n_fit: Any) -> Result:
        """extrapolate() through the cache; the fit's own exceptions pass through unchanged."""
        key = None
        if self.mode != "off":
            try:
                key = self._key(x_cells, y_cells, model_type, b_fixed, n_fit)
            except Exception:   # _Uncacheable, or a fingerprint problem
                self.uncacheable += 1
        if key is None:
            return _extrapolate_quietly(x_cells, y_cells, model_type, b_fixed, n_fit)

        with self._lock:
            result = self._memory.get(key)
            if result is not None:
                self._memory.move_to_end(key)
                self.hits_memory += 1
                return result
        result = self.disk.get(key) if self.disk is not None else None
        if result is not None:
            self.hits_disk += 1
        else:
            self.misses += 1
            result = _extrapolate_quietly(x_cells, y_cells, model_type, b_fixed, n_fit)
            if self.disk is not None:
                self.disk.put(key, result)
        with self._lock:
            self._memory[key] = result
            while len(self._memory) > _MEMORY_ENTRIES:
                self._memory.popitem(last=False)
        return result

    def stats(self) -> str:
        if self.mode == "off":
            return "cache off (EXTRAPLUS_CACHE=off)"
        if self.disk is not None:
            rows = self.disk.count()
            disk = f"file {self.disk.path} ({rows} results, {self.disk.state})" if rows is not None \
                else f"file {self.disk.path} ({self.disk.state})"
        elif self._disk_path is not None:
            disk = f"file {self._disk_path} (not opened yet)"
        else:
            disk = "no file"
        return (f"cache {self.mode}; memory {len(self._memory)} results; {disk}; "
                f"hits {self.hits_memory} memory + {self.hits_disk} file, "
                f"misses {self.misses}, not cacheable {self.uncacheable}")


_CACHE = _ResultCache()


def _error_text(exc: Exception) -> str:
    """ValueError messages come from the extraplus input checks and are shown as
    they are; any other exception also shows its type."""
    if isinstance(exc, ValueError):
        return f"Error: {exc}"
    return f"Error: {type(exc).__name__}: {exc}"


def _excel_errors(function=None, *, shape=None):
    """
    Return a short "Error: ..." text to Excel instead of raising.

    Only the failure path changes: when the wrapped UDF succeeds, its value is
    returned unchanged.  With ``shape`` the text fills an array of that shape,
    the shape of the function's normal output: xlwings resizes an
    expand="table" formula to whatever it returns, so a single text value
    would clear the other cells of the array (and the B cell that other
    formulas read as b_fixed).
    """

    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:  # never hand a traceback to an Excel cell
                text = _error_text(exc)
                return text if shape is None else np.full(shape, text, dtype=object)

        return wrapper

    return decorate if function is None else decorate(function)


# ---------------------------------------------------------------------------
# Excel plot helpers
# ---------------------------------------------------------------------------


def _insert_plot_in_excel(
    caller: Any,
    fig: Figure,
    picture_name: str,
) -> None:
    """
    Insert/update the Matplotlib figure directly below the Excel formula cell.

    The picture name includes the calling cell, so several EXTRAP*_PLOT
    formulas can coexist on the same worksheet.  update=True means that
    recalculation replaces the existing picture instead of creating duplicates.
    """
    if caller is None:
        raise RuntimeError(
            "This plot function must be called from an Excel worksheet through xlwings."
        )

    safe_name = (
        f"Extraplus_{picture_name}_R{int(caller.row)}_C{int(caller.column)}"
    )

    caller.sheet.pictures.add(
        fig,
        name=safe_name,
        update=True,
        left=caller.left,
        top=caller.top + caller.height,
    )


def _run_plot_udf(
    x_range: Any,
    y_range: Any,
    model_type: str,
    caller: Any,
    picture_name: str,
    log_plot: bool,
    b_fixed: Any = None,
    n_fit: Any = None,
) -> str:
    """Shared implementation for all six Excel plot UDFs."""
    result = _extrapolate_quietly(x_range, y_range, model_type, b_fixed, n_fit)
    fig = result.plot_linearized() if log_plot else result.plot()

    try:
        _insert_plot_in_excel(
            caller=caller,
            fig=fig,
            picture_name=picture_name,
        )
    finally:
        plt.close(fig)

    return f"{picture_name} updated"


# ---------------------------------------------------------------------------
# Extrapolated values at infinity
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).value


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).value


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def EXTRAP3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law extrapolated value C at infinity."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).value


# ---------------------------------------------------------------------------
# Fitted values at a finite basis size
# ---------------------------------------------------------------------------

# target defaults to None so that Excel calls Python (and shows "Error: target
# is required.") when it is left out, instead of rejecting the formula.


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP1(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Exponential fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).predict(target)


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP2(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Square-root-exponential fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).predict(target)


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("target", doc="Basis size to evaluate the fit at, usually beyond the data (a number > 0).")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.", err_to_str=True)
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.", err_to_str=True)
@_excel_errors
def FINITE_EXTRAP3(x_range, y_range, target=None, b_fixed=None, n_fit=None):
    """Power-law fit evaluated at the basis size target."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).predict(target)


# ---------------------------------------------------------------------------
# Uncertainty values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def UNCERTAINTY1(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def UNCERTAINTY2(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential asymptote uncertainty in original y units."""
    return _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    ).uncertainty


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def UNCERTAINTY3(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law asymptote uncertainty in the original y units."""
    return _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    ).uncertainty


# ---------------------------------------------------------------------------
# Fitted B values
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE1(x_range, y_range, b_fixed=None, n_fit=None):
    """Exponential B for the normalized variable x/x_max."""
    return _run_udf(x_range, y_range, _EXPONENTIAL, b_fixed, n_fit).decay_rate


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE2(x_range, y_range, b_fixed=None, n_fit=None):
    """Square-root-exponential B for sqrt(x/x_max)."""
    return _run_udf(x_range, y_range, _SQRT_EXPONENTIAL, b_fixed, n_fit).decay_rate


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors
def DECAY_RATE3(x_range, y_range, b_fixed=None, n_fit=None):
    """Power-law B for (x/x_max)^(-B)."""
    return _run_udf(x_range, y_range, _POWER_LAW, b_fixed, n_fit).decay_rate


# ---------------------------------------------------------------------------
# Combined spill functions: C, uncertainty, B
# ---------------------------------------------------------------------------


def _vertical_fit_output(result: Result):
    """Return C, uncertainty, and B as an explicit 3 x 1 Excel array."""
    return np.asarray(
        [
            [result.value],
            [result.uncertainty],
            [result.decay_rate],
        ],
        dtype=float,
    )


def _horizontal_fit_output(result: Result):
    """Return C, uncertainty, and B as an explicit 1 x 3 Excel array."""
    return np.asarray(
        [[result.value, result.uncertainty, result.decay_rate]],
        dtype=float,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
def EXTRAP1_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
def EXTRAP2_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(3, 1))
def EXTRAP3_V(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law fit. Spills C, uncertainty, and B vertically."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    )
    return _vertical_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
def EXTRAP1_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
def EXTRAP2_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Square-root-exponential fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _SQRT_EXPONENTIAL,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


@xw.func(category="Extrapolation", volatile=False)
@xw.ret(expand="table")
@xw.arg("x_range", np.array, ndim=2)
@xw.arg("y_range", np.array, ndim=2)
@xw.arg("b_fixed", err_to_str=True)
@xw.arg("n_fit", err_to_str=True)
@_excel_errors(shape=(1, 3))
def EXTRAP3_H(
    x_range,
    y_range,
    b_fixed=None,
    n_fit=None,
):
    """Power-law fit. Spills C, uncertainty, and B horizontally."""
    result = _run_udf(
        x_range,
        y_range,
        _POWER_LAW,
        b_fixed,
        n_fit,
    )
    return _horizontal_fit_output(result)


# ---------------------------------------------------------------------------
# All three models in one call
# ---------------------------------------------------------------------------


def _b_fixed_per_model(b_fixed: Any) -> tuple:
    """b_fixed of EXTRAP_ALL_V: blank (fit every B) or three cells, one B per model."""
    if b_fixed is None:
        return (None, None, None)
    cells = _cells(b_fixed)
    if len(cells) == 1 and (cells[0] is None or cells[0] == ""):
        return (None, None, None)
    if len(cells) != 3:
        raise ValueError("b_fixed must be blank or 3 cells, one B for each model (1, 2, 3).")
    return cells


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", ndim=2, err_to_str=True,   # not np.array: it reads blank cells as NaN
        doc="Optional: 3 cells with a fixed B for models 1, 2, 3 (a blank cell fits that B). Leave out to fit all.")
@_excel_errors(shape=(3, 3))
def EXTRAP_ALL_V(x_range, y_range, b_fixed=None):
    """All three models: columns 1-3 are EXTRAP1_V-EXTRAP3_V (rows C, uncertainty, B)."""
    columns = []
    for model, b in zip((_EXPONENTIAL, _SQRT_EXPONENTIAL, _POWER_LAW), _b_fixed_per_model(b_fixed)):
        try:
            columns.append(_vertical_fit_output(_run_udf(x_range, y_range, model, b)))
        except Exception as exc:   # this model only: the others are still shown
            columns.append(np.full((3, 1), _error_text(exc), dtype=object))
    return np.hstack(columns)


# ---------------------------------------------------------------------------
# Installed version
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@_excel_errors
def EXTRAPLUS_VERSION():
    """Installed version of the extraplus package."""
    return extraplus.__version__


@xw.func(category="Extrapolation", volatile=False)
@_excel_errors
def EXTRAPLUS_CACHE_STATS():
    """State of the result cache: mode, file, hits and misses (Ctrl+Alt+F9 refreshes it)."""
    return _CACHE.stats()


# ---------------------------------------------------------------------------
# Write values: replace Extraplus formulas by the numbers they compute
# ---------------------------------------------------------------------------
#
# Run from the add-in's ribbon button (RunPython in the UDF server):
#
#     extrapolation.write_values("'Sheet'!$A$1:$Z$99")
#
# For every Extraplus formula in the range (or on the whole sheet), whose
# whole text is one call such as =EXTRAP1_V($A$2:$A$13,B2:B13,BV4):
#   - the arguments are read from the sheet with the same xlwings options the
#     function itself gets, and the same Python function computes the result,
#     so the numbers are those the formula gives;
#   - a b_fixed (or other) argument inside another such formula's cells is
#     handled after that formula has been written, from the written numbers;
#     if that formula is not written, neither is this one;
#   - the numbers replace the formula only if they fill exactly the formula's
#     cells and are not an "Error: ..." result.
# Before the first write a copy of the workbook is saved next to it, and each
# written block is recorded on the sheet "Extraplus values" (formula, inputs
# digest, version).  Anything else (nested formulas, names, other workbooks)
# is left as it is and listed.  Calculation, events and screen updating are
# set for the run and restored afterwards.

_VALUE_FUNCTIONS = frozenset(
    [f"{base}{n}" for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE", "FINITE_EXTRAP") for n in (1, 2, 3)]
    + [f"EXTRAP{n}{suffix}" for suffix in ("_V", "_H") for n in (1, 2, 3)]
    + ["EXTRAP_ALL_V"]
)
_RECORD_SHEET = "Extraplus values"

# The read options of each argument, as registered with xlwings (checked
# against the registration by tests/test_registration.py).  On Windows the
# registered options themselves are used.
_RANGE = {"convert": np.array, "ndim": 2}
_ERRORS_AS_TEXT = {"err_to_str": True}


def _fallback_arg_options(func_name: str) -> list:
    if func_name == "EXTRAP_ALL_V":
        return [("x_range", _RANGE), ("y_range", _RANGE), ("b_fixed", {"ndim": 2, **_ERRORS_AS_TEXT})]
    args = [("x_range", _RANGE), ("y_range", _RANGE)]
    if func_name.startswith("FINITE_"):
        args.append(("target", {}))
    return args + [("b_fixed", _ERRORS_AS_TEXT), ("n_fit", _ERRORS_AS_TEXT)]


def _arg_options(func_name: str) -> list:
    """[(name, xlwings read options)] for the function's arguments, in order."""
    info = getattr(globals()[func_name], "__xlfunc__", None)
    if info is None:
        return _fallback_arg_options(func_name)
    return [(arg["name"], dict(arg["options"])) for arg in info["args"]]


_COL = r"\$?[A-Za-z]{1,3}"
_ROW = r"\$?[0-9]{1,7}"
_CELL = _COL + _ROW
_AREA = rf"(?:{_CELL}(?::{_CELL})?|{_COL}:{_COL}|{_ROW}:{_ROW})"
_SHEET = r"(?:'(?:[^']|'')+'|[A-Za-z_][\w.]*)"
_REF_RE = re.compile(rf"^(?:(?P<sheet>{_SHEET})!)?(?P<area>{_AREA})$")
_NUMBER_RE = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_CALL_RE = re.compile(
    r"^=\s*(?:'[^']*extraplus\.xlam'!|extraplus\.xlam!)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL)


def _split_args(text: str) -> Optional[list]:
    """Split a formula's argument text at top-level commas; None if it has nested calls or strings."""
    args, current, quoted = [], [], False
    for char in text:
        if quoted:
            current.append(char)
            if char == "'":
                quoted = False   # '' re-enters the quote on the next character
        elif char == "'":
            quoted = True
            current.append(char)
        elif char in '("{}':
            return None
        elif char == ",":
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quoted:
        return None
    args.append("".join(current).strip())
    return args


def _column_number(letters: str) -> int:
    number = 0
    for char in letters.upper():
        number = number * 26 + ord(char) - 64
    return number


def _area_box(area: str) -> tuple:
    """(row1, col1, row2, col2) of an A1 area; whole rows/columns use 1..max."""
    parts = area.replace("$", "").split(":")
    boxes = []
    for part in parts:
        letters = "".join(c for c in part if c.isalpha())
        digits = "".join(c for c in part if c.isdigit())
        boxes.append((int(digits) if digits else None, _column_number(letters) if letters else None))
    (r1, c1), (r2, c2) = boxes[0], boxes[-1]
    return (r1 or 1, c1 or 1, r2 or 1_048_576, c2 or 16_384)


def _sheet_name(text: Optional[str], default: str) -> str:
    if not text:
        return default
    return text[1:-1].replace("''", "'") if text.startswith("'") else text


def _parse_call(formula: str, sheet: str) -> Optional[tuple]:
    """(function name, [argument]) for a formula that is exactly one Extraplus call, else None.

    An argument is ("missing",), ("number", text) or ("ref", sheet, area, box).
    """
    match = _CALL_RE.match(formula or "")
    if not match or match.group("name").upper() not in _VALUE_FUNCTIONS:
        return None
    name = match.group("name").upper()
    texts = _split_args(match.group("args"))
    if texts is None or len(texts) > len(_fallback_arg_options(name)):
        return None
    args = []
    for text in texts:
        if text == "":
            args.append(("missing",))
        elif _NUMBER_RE.match(text):
            args.append(("number", text))
        else:
            ref = _REF_RE.match(text)
            if not ref:
                return None
            ref_sheet = _sheet_name(ref.group("sheet"), sheet)
            args.append(("ref", ref_sheet, ref.group("area"), _area_box(ref.group("area"))))
    if len(args) < 2 or any(a[0] != "ref" for a in args[:2]):
        return None
    return name, args


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _result_shape(value: Any) -> tuple:
    return value.shape if isinstance(value, np.ndarray) else (1, 1)


def _is_error_result(value: Any) -> bool:
    cells = value.ravel() if isinstance(value, np.ndarray) else [value]
    return any(isinstance(v, str) and v.startswith("Error: ") for v in cells)


def _same_number(a: Any, b: Any) -> bool:
    if isinstance(a, (float, int, np.floating)) and isinstance(b, (float, int, np.floating)):
        return struct.pack("<d", float(a)) == struct.pack("<d", float(b))
    return a == b


def _values_pass(book, module=None) -> dict:
    """
    The write-values logic on a workbook adapter (see _XlwingsBook for the
    real one and tests/test_write_values.py for a fake).  Returns the report.
    """
    module = module or sys.modules[__name__]
    blocks, unrecognized = [], []
    covered = set()
    for sheet, row, col, formula in book.formula_cells():
        if (sheet, row, col) in covered:
            continue
        box = book.block_of(sheet, row, col)
        covered.update((sheet, r, c) for r in range(box[0], box[2] + 1) for c in range(box[1], box[3] + 1))
        call = _parse_call(formula, sheet)
        if call is None:
            unrecognized.append((sheet, box, formula))
            continue
        blocks.append({"sheet": sheet, "box": box, "formula": formula, "name": call[0], "args": call[1],
                       "deps": set(), "status": None, "reason": ""})

    # Order: a block that reads cells of another block comes after it.
    for i, block in enumerate(blocks):
        for arg in block["args"]:
            if arg[0] != "ref":
                continue
            for j, other in enumerate(blocks):
                if j != i and other["sheet"] == arg[1] and _overlaps(other["box"], arg[3]):
                    block["deps"].add(j)
    order, done, visiting = [], set(), set()

    def visit(i):
        if i in done:
            return True
        if i in visiting:
            return False
        visiting.add(i)
        ok = all(visit(j) for j in blocks[i]["deps"])
        visiting.discard(i)
        if ok:
            done.add(i)
            order.append(i)
        else:
            blocks[i]["status"], blocks[i]["reason"] = "skipped", "circular reference"
        return ok

    for i in range(len(blocks)):
        visit(i)

    report = {"written": 0, "failed": 0, "skipped": 0, "unrecognized": len(unrecognized),
              "changed": 0, "backup": None, "problems": [], "stopped": None}
    records = []
    try:
        with book.calculation_paused():
            _write_blocks(book, module, blocks, order, report, records)
    except Exception as exc:   # Excel refused a read or write: keep what is done, say so
        report["stopped"] = _error_text(exc)
    for block in blocks:
        report[block["status"] or "skipped"] += 1
        if block["status"] != "written":
            report["problems"].append((book.address(block["sheet"], block["box"]), block["status"] or "skipped",
                                       block["reason"] or "not reached"))
    for sheet, box, formula in unrecognized:
        report["problems"].append((book.address(sheet, box), "left as formula",
                                   "not a single Extraplus call with plain cell references"))
    if records:
        book.record(records, report)
    return report


def _write_blocks(book, module, blocks, order, report, records) -> None:
    for i in order:
        block = blocks[i]
        if block["status"]:
            continue
        blocked = [blocks[j] for j in block["deps"] if blocks[j]["status"] != "written"]
        if blocked:
            block["status"] = "skipped"
            block["reason"] = f"uses {book.address(blocked[0]['sheet'], blocked[0]['box'])}, which was not written"
            continue
        # A NaN result is written as an empty cell, and an empty b_fixed means "fit B".
        empty = [blocks[j] for j in block["deps"] if _reads_empty_output(block, blocks[j])]
        if empty:
            block["status"] = "skipped"
            block["reason"] = f"uses an empty (NaN) result of {book.address(empty[0]['sheet'], empty[0]['box'])}"
            continue
        values = []
        for (_name, options), arg in zip(_arg_options(block["name"]), block["args"]):
            if arg[0] == "missing":
                values.append(None)
            elif arg[0] == "number":
                values.append(book.convert(book.evaluate(arg[1]), options))
            else:
                values.append(book.read(arg[1], arg[2], options))
        result = getattr(module, block["name"])(*values)
        shape = (block["box"][2] - block["box"][0] + 1, block["box"][3] - block["box"][1] + 1)
        if _is_error_result(result):
            block["status"] = "failed"
            block["reason"] = str(result.ravel()[0] if isinstance(result, np.ndarray) else result)
            continue
        if _result_shape(result) != shape:
            block["status"], block["reason"] = "skipped", (
                f"its result is {_result_shape(result)[0]} x {_result_shape(result)[1]} cells, "
                f"the formula fills {shape[0]} x {shape[1]}")
            continue
        if report["backup"] is None:
            report["backup"] = book.backup()
        before = np.asarray(book.values(block["sheet"], block["box"]), dtype=object).ravel()
        after = np.asarray(result, dtype=object).reshape(shape)
        if not all(_same_number(a, b) for a, b in zip(before, after.ravel())):
            report["changed"] += 1
        book.write(block["sheet"], block["box"], result)
        block["status"], block["written"] = "written", after
        records.append((book.address(block["sheet"], block["box"]), block["formula"], block["name"],
                        _inputs_digest(values),
                        ", ".join(repr(float(v)) if isinstance(v, (float, np.floating)) else repr(v)
                                  for v in after.ravel())))


def _reads_empty_output(block: dict, upstream: dict) -> bool:
    """Whether an argument of block reads a cell of upstream's written result that is NaN or None."""
    top, left = upstream["box"][0], upstream["box"][1]
    for arg in block["args"]:
        if arg[0] != "ref" or arg[1] != upstream["sheet"] or not _overlaps(arg[3], upstream["box"]):
            continue
        for r in range(max(arg[3][0], upstream["box"][0]), min(arg[3][2], upstream["box"][2]) + 1):
            for c in range(max(arg[3][1], upstream["box"][1]), min(arg[3][3], upstream["box"][3]) + 1):
                v = upstream["written"][r - top, c - left]
                if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                    return True
    return False


def _inputs_digest(values: list) -> str:
    """A short SHA-256 of the exact argument values, to tell later whether the inputs changed."""
    out = bytearray(_CACHE_SCHEMA)
    for value in values:
        try:
            _encode(None if value is None else tuple(_cells(value)), out)
        except _Uncacheable:
            out += repr(value).encode()
    return hashlib.sha256(out).hexdigest()[:16]

def _summary(report: dict) -> str:
    lines = [f"Written: {report['written']} formulas replaced by their values."]
    if report["stopped"]:
        lines.insert(0, f"Stopped early: {report['stopped']}")
    if report["changed"]:
        lines.append(f"{report['changed']} of them showed different values before (e.g. errors or "
                     "values not recalculated).")
    for key, label in (("failed", "kept (the function returns an error)"),
                       ("skipped", "kept (see the list)"), ("unrecognized", "kept (formula not recognized)")):
        if report[key]:
            lines.append(f"{report[key]} {label}.")
    if report["backup"]:
        lines.append(f"Backup of the workbook before the change: {report['backup']}")
    if report["problems"]:
        lines.append("")
        lines += [f"{where}: {status} - {why}" for where, status, why in report["problems"][:15]]
        if len(report["problems"]) > 15:
            lines.append(f"... and {len(report['problems']) - 15} more (see the sheet '{_RECORD_SHEET}').")
    return "\n".join(lines)


_XL_CALCULATION_MANUAL = -4135
_XL_CELL_TYPE_FORMULAS = -4123
_MENTIONS_RE = re.compile(r"\b(?:" + "|".join(sorted(_VALUE_FUNCTIONS, key=len, reverse=True)) + r")\s*\(",
                          re.IGNORECASE)


def _column_letters(number: int) -> str:
    letters = ""
    while number:
        number, rest = divmod(number - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


class _XlwingsBook:
    """The workbook side of _values_pass, through xlwings (Windows, inside Excel)."""

    def __init__(self, book, scope):
        self.book, self.app, self.scope = book, book.app, scope

    def formula_cells(self):
        name = self.scope.sheet.name
        try:
            found = self.scope.api.SpecialCells(_XL_CELL_TYPE_FORMULAS)
        except Exception:   # Excel raises when there are no formula cells
            return
        for area in found.Areas:
            row0, col0, formulas = area.Row, area.Column, area.Formula
            rows = formulas if isinstance(formulas, tuple) else ((formulas,),)
            for i, row in enumerate(rows):
                for j, text in enumerate(row):
                    if isinstance(text, str) and _MENTIONS_RE.search(text):
                        yield name, row0 + i, col0 + j, text

    def block_of(self, sheet, row, col):
        cell = self.book.sheets[sheet].api.Cells(row, col)
        block = None
        if cell.HasArray:
            block = cell.CurrentArray
        else:
            try:
                if cell.HasSpill:
                    block = cell.SpillingToRange
            except Exception:   # no dynamic arrays in this Excel
                pass
        if block is None:
            return (row, col, row, col)
        return (block.Row, block.Column, block.Row + block.Rows.Count - 1, block.Column + block.Columns.Count - 1)

    def _range(self, sheet, box):
        return self.book.sheets[sheet].range((box[0], box[1]), (box[2], box[3]))

    def address(self, sheet, box):
        first = f"{_column_letters(box[1])}{box[0]}"
        last = f"{_column_letters(box[3])}{box[2]}"
        return f"'{sheet}'!{first}" + ("" if first == last else f":{last}")

    def read(self, sheet, area, options):
        # What xlwings does for a range argument: conversion.read(range, None, options).
        return self.book.sheets[sheet].range(area).options(**options).value

    def evaluate(self, text):
        return self.app.api.Evaluate(text)

    def convert(self, value, options):
        # What xlwings does for a value argument (a number typed in the formula).
        from xlwings import conversion
        return conversion.read(None, value, options)

    def values(self, sheet, box):
        return self._range(sheet, box).options(ndim=2).value

    def write(self, sheet, box, result):
        target = self._range(sheet, box)
        try:
            target.value = result
        except Exception:   # e.g. Excel refuses to overwrite an array formula cell by cell
            target.api.ClearContents()
            target.value = result

    def backup(self):
        full = self.book.fullname
        if not self.book.api.Path or "://" in full:
            raise RuntimeError("the workbook must be saved in a local folder first (a backup copy is made next to it)")
        path = Path(full)
        stamp = time.strftime("%Y-%m-%d %H%M%S")
        copy = path.with_name(f"{path.stem} (before Extraplus values {stamp}){path.suffix}")
        self.book.api.SaveCopyAs(str(copy))
        return str(copy)

    def calculation_paused(self):
        app = self.app.api

        class _Paused:
            def __enter__(inner):
                inner.saved = (app.Calculation, app.EnableEvents, app.ScreenUpdating)
                app.Calculation = _XL_CALCULATION_MANUAL
                app.EnableEvents = False
                app.ScreenUpdating = False

            def __exit__(inner, *exc):
                for name, value in zip(("Calculation", "EnableEvents", "ScreenUpdating"), inner.saved):
                    try:
                        if getattr(app, name) != value:
                            setattr(app, name, value)
                    except Exception:
                        pass
                return False

        return _Paused()

    def record(self, records, report):
        sheets = self.book.sheets
        if _RECORD_SHEET in [s.name for s in sheets]:
            sheet = sheets[_RECORD_SHEET]
        else:
            sheet = sheets.add(_RECORD_SHEET, after=sheets[-1])
            sheet.range("A1").value = [["Time", "Range", "Formula (as it was)", "Function",
                                        "Inputs SHA-256", "Values written / note"]]
        used = sheet.api.UsedRange
        next_row = used.Row + used.Rows.Count
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fingerprint = _code_fingerprint().hex()[:12]
        rows = [["run", "", f"extraplus {extraplus.__version__} ({fingerprint})", "",
                 _summary({**report, "problems": []}).replace("\n", " ")]]
        rows += [[where, formula, name, digest, written] for where, formula, name, digest, written in records]
        rows += [[where, "", status, "", why] for where, status, why in report["problems"]]
        # "'" keeps every entry as text: formulas, addresses starting with ', digits.
        sheet.range((next_row, 1)).value = [[stamp] + ["'" + str(v) if v else "" for v in row] for row in rows]


def write_values() -> None:
    """The ribbon command: Extraplus formulas of the selection (or of the active sheet) to values."""
    book = xw.Book.caller()
    title = "Extraplus: write values"
    try:
        selection = book.app.selection
        scope = selection if selection is not None and selection.count > 1 else book.sheets.active.used_range
        report = _values_pass(_XlwingsBook(book, scope))
    except Exception as exc:
        book.app.alert(f"Nothing was written: {_error_text(exc)}", title=title, mode="critical")
        return
    book.app.alert(_summary(report), title=title, mode="critical" if report["stopped"] else "info")


# ---------------------------------------------------------------------------
# Matplotlib plots in Excel
# ---------------------------------------------------------------------------


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP1_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the exponential model and insert extrap.py's standard Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP1_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP2_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the square-root-exponential model and insert extrap.py's standard
    Matplotlib plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_SQRT_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP2_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP3_PLOT(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the power-law model and insert extrap.py's standard Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_POWER_LAW,
        caller=caller,
        picture_name="EXTRAP3_PLOT",
        log_plot=False,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP1_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the exponential model and insert extrap.py's logarithmic Matplotlib
    plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP1_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP2_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the square-root-exponential model and insert extrap.py's logarithmic
    Matplotlib plot directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_SQRT_EXPONENTIAL,
        caller=caller,
        picture_name="EXTRAP2_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )


@xw.func(category="Extrapolation", volatile=False)
@xw.arg("x_range", np.array, ndim=2, doc="Range containing positive basis-size x values.")
@xw.arg("y_range", np.array, ndim=2, doc="Range containing the corresponding y values.")
@xw.arg("b_fixed", doc="Optional positive fixed B. Leave blank to fit B.")
@xw.arg("n_fit", doc="Optional number of first numeric rows used in the fit.")
def EXTRAP3_PLOT_LOG(
    x_range,
    y_range,
    caller,
    b_fixed=None,
    n_fit=None,
):
    """
    Fit the power-law model and insert extrap.py's logarithmic Matplotlib plot
    directly below the calling Excel cell.
    """
    return _run_plot_udf(
        x_range=x_range,
        y_range=y_range,
        model_type=_POWER_LAW,
        caller=caller,
        picture_name="EXTRAP3_PLOT_LOG",
        log_plot=True,
        b_fixed=b_fixed,
        n_fit=n_fit,
    )



__all__ = [
    "EXTRAP1",
    "EXTRAP2",
    "EXTRAP3",
    "FINITE_EXTRAP1",
    "FINITE_EXTRAP2",
    "FINITE_EXTRAP3",
    "UNCERTAINTY1",
    "UNCERTAINTY2",
    "UNCERTAINTY3",
    "DECAY_RATE1",
    "DECAY_RATE2",
    "DECAY_RATE3",
    "EXTRAP1_V",
    "EXTRAP2_V",
    "EXTRAP3_V",
    "EXTRAP1_H",
    "EXTRAP2_H",
    "EXTRAP3_H",
    "EXTRAP_ALL_V",
    "EXTRAPLUS_VERSION",
    "EXTRAPLUS_CACHE_STATS",
    "EXTRAP1_PLOT",
    "EXTRAP2_PLOT",
    "EXTRAP3_PLOT",
    "EXTRAP1_PLOT_LOG",
    "EXTRAP2_PLOT_LOG",
    "EXTRAP3_PLOT_LOG",
]


# Performance investigation only: off unless EXTRAPLUS_PROFILE is set in Excel's
# environment. It wraps references to time the UDF path; results are unchanged.
if os.environ.get("EXTRAPLUS_PROFILE"):
    try:
        import extraplus_profile
    except ImportError:   # profiling module not shipped with this install: run unprofiled
        extraplus_profile = None
    if extraplus_profile is not None:
        extraplus_profile.install(sys.modules[__name__])