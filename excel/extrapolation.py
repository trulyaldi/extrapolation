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
@xw.arg("b_fixed", np.array, ndim=2, err_to_str=True,
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