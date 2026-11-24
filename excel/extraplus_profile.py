"""
Opt-in profiling of the Extraplus UDF path (performance investigation only).

extrapolation.py imports this module only when the environment variable
EXTRAPLUS_PROFILE is set (in the Excel process, so the UDF server inherits it).
When the variable is unset, nothing here runs and nothing is wrapped.

Levels (value of EXTRAPLUS_PROFILE):
  1  timings and counters, aggregated (no per-call log lines, no extra COM calls)
  2  as 1, plus the caller's sheet!address per call, to count calls per formula
     (costs extra COM calls per UDF call; measure it before trusting level-2 timings)

Output: EXTRAPLUS_PROFILE_DIR (default: %TEMP%) / extraplus-profile-<pid>.json,
rewritten at most every 0.5 s while counters change. Counters are cumulative for
the life of the UDF server, so a run is measured as the difference between two
snapshots.

What is wrapped (references only; arguments and return values pass through untouched):
  xlwings.com_server.XLPython.CallUDF   whole Python side of one UDF call, and the
                                        gap between calls (time outside Python)
  xlwings.udfs.get_udf_module           module lookup (stat of the .py file)
  xlwings.conversion.read / write       argument and return conversion (COM reads of .Value etc.)
  xlwings.udfs.get_cache_key            expand="table" cache key (COM reads of the caller)
  xlwings.udfs.delayed_resize_dynamic_array_formula   counts resizes
  xlwings._xlwindows COMRetry*          counts COM operations made through xlwings wrappers
  extrapolation._cells, _run_udf, _CACHE.extrapolate, _extrapolate_quietly
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict
from functools import wraps

_perf = time.perf_counter

_lock = threading.Lock()
_state = {
    "installed_xlwings": False,
    "module": None,
}
_local = threading.local()

# Cumulative counters.
_time = defaultdict(float)        # seconds per segment
_count = defaultdict(int)         # calls per segment / event
_b_kind = defaultdict(int)        # how the b_fixed argument arrived
_per_address = defaultdict(int)   # level 2: calls per sheet!address
_gaps = {"n": 0, "sum": 0.0, "max": 0.0, "idle_n": 0}
_last_exit = [None]
_version = [0]
_GAP_IDLE_S = 5.0                 # a gap longer than this is idle time, not per-call overhead


def _level() -> int:
    try:
        return int(os.environ.get("EXTRAPLUS_PROFILE", "0") or "0")
    except ValueError:
        return 1


def _timed(name, function):
    """Wrap function so its wall time and call count are added to segment `name`."""
    if getattr(function, "_extraplus_profiled", False):
        return function

    @wraps(function)
    def wrapper(*args, **kwargs):
        t0 = _perf()
        try:
            return function(*args, **kwargs)
        finally:
            dt = _perf() - t0
            with _lock:
                _time[name] += dt
                _count[name] += 1
                _version[0] += 1

    wrapper._extraplus_profiled = True
    wrapper._extraplus_original = function
    return wrapper


def _counted(name, function):
    if getattr(function, "_extraplus_profiled", False):
        return function

    @wraps(function)
    def wrapper(*args, **kwargs):
        with _lock:
            _count[name] += 1
            _version[0] += 1
        return function(*args, **kwargs)

    wrapper._extraplus_profiled = True
    wrapper._extraplus_original = function
    return wrapper


def _classify_b(b) -> str:
    if b is None:
        return "none"
    if isinstance(b, bool):
        return "bool"
    if isinstance(b, (int, float)):
        return "nan" if (isinstance(b, float) and math.isnan(b)) else "number"
    if isinstance(b, str):
        return "error_text" if b.startswith("#") else "text"
    return type(b).__name__


# ---------------------------------------------------------------------------
# xlwings hooks (Windows UDF server only)
# ---------------------------------------------------------------------------


def _install_xlwings(level: int) -> None:
    if _state["installed_xlwings"]:
        return
    try:
        import xlwings.com_server as com_server
        import xlwings.udfs as udfs
        import xlwings.conversion as conversion
        import xlwings._xlwindows as xlwindows
    except Exception:          # not on Windows, or xlwings missing: profile our code only
        return

    original_call = com_server.XLPython.CallUDF

    def CallUDF(self, script, fname, args, this_workbook=None, caller=None):
        t0 = _perf()
        _local.com_ops = 0
        with _lock:
            last = _last_exit[0]
            if last is not None:
                gap = t0 - last
                if gap <= _GAP_IDLE_S:
                    _gaps["n"] += 1
                    _gaps["sum"] += gap
                    _gaps["max"] = max(_gaps["max"], gap)
                else:
                    _gaps["idle_n"] += 1
        address = None
        if level >= 2 and caller is not None:
            t1 = _perf()
            try:
                c = com_server.FromVariant(caller)
                address = f"{c.Worksheet.Name}!{c.Address}"
            except Exception:
                address = "?"
            with _lock:
                _time["level2_address"] += _perf() - t1
        try:
            return original_call(self, script, fname, args, this_workbook, caller)
        finally:
            t_end = _perf()
            with _lock:
                _time["CallUDF"] += t_end - t0
                _count["CallUDF"] += 1
                _count["CallUDF:" + str(fname)] += 1
                _count["com_ops_in_calls"] += getattr(_local, "com_ops", 0)
                if address is not None:
                    _per_address[address] += 1
                _last_exit[0] = _perf()
                _version[0] += 1
            _local.com_ops = None

    CallUDF._extraplus_profiled = True
    com_server.XLPython.CallUDF = CallUDF

    udfs.get_udf_module = _timed("xlwings.get_udf_module", udfs.get_udf_module)
    udfs.get_cache_key = _timed("xlwings.get_cache_key", udfs.get_cache_key)
    udfs.delayed_resize_dynamic_array_formula = _counted(
        "xlwings.resize_scheduled", udfs.delayed_resize_dynamic_array_formula)
    conversion.read = _timed("xlwings.conversion.read", conversion.read)
    conversion.write = _timed("xlwings.conversion.write", conversion.write)

    def _count_com(cls, attr):
        original = getattr(cls, attr)

        def wrapper(*args, **kwargs):
            n = getattr(_local, "com_ops", None)
            if n is not None:
                _local.com_ops = n + 1
            else:
                with _lock:
                    _count["com_ops_outside_calls"] += 1
            return original(*args, **kwargs)

        setattr(cls, attr, wrapper)

    _count_com(xlwindows.COMRetryObjectWrapper, "__getattr__")
    _count_com(xlwindows.COMRetryObjectWrapper, "__setattr__")
    _count_com(xlwindows.COMRetryMethodWrapper, "__call__")
    _state["installed_xlwings"] = True


# ---------------------------------------------------------------------------
# Our module
# ---------------------------------------------------------------------------


def _install_module(module) -> None:
    module._cells = _timed("ours._cells", module._cells)
    module._extrapolate_quietly = _timed("ours.fit(_extrapolate_quietly)", module._extrapolate_quietly)

    original_run = module._run_udf

    @wraps(original_run)
    def _run_udf(x_range, y_range, model_type, b_fixed=None, n_fit=None):
        kind = _classify_b(b_fixed)
        with _lock:
            _b_kind[kind] += 1
        t0 = _perf()
        try:
            return original_run(x_range, y_range, model_type, b_fixed, n_fit)
        finally:
            with _lock:
                _time["ours._run_udf"] += _perf() - t0
                _count["ours._run_udf"] += 1
                _version[0] += 1

    _run_udf._extraplus_profiled = True
    module._run_udf = _run_udf

    cache = module._CACHE
    cache.extrapolate = _timed("ours.cache.extrapolate", cache.extrapolate)
    _state["module"] = module


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot() -> dict:
    with _lock:
        data = {
            "pid": os.getpid(),
            "time": time.time(),
            "level": _level(),
            "seconds": dict(_time),
            "counts": dict(_count),
            "b_fixed_kind": dict(_b_kind),
            "gaps": dict(_gaps),
            "per_address": dict(_per_address),
        }
    module = _state["module"]
    if module is not None:
        c = module._CACHE
        data["cache"] = {"mode": c.mode, "hits_memory": c.hits_memory, "hits_disk": c.hits_disk,
                         "misses": c.misses, "uncacheable": c.uncacheable}
    return data


def _output_path() -> str:
    folder = os.environ.get("EXTRAPLUS_PROFILE_DIR") or tempfile.gettempdir()
    return os.path.join(folder, f"extraplus-profile-{os.getpid()}.json")


def write_snapshot() -> str:
    path = _output_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(snapshot(), fh)
    os.replace(tmp, path)
    return path


def _writer() -> None:
    written = -1
    while True:
        time.sleep(0.5)
        v = _version[0]
        if v != written:
            try:
                write_snapshot()
                written = v
            except Exception:
                pass


def install(module) -> None:
    """Called by extrapolation.py at import (and again if xlwings reloads it)."""
    level = _level()
    _install_xlwings(level)
    _install_module(module)
    if not _state.get("writer"):
        t = threading.Thread(target=_writer, name="extraplus-profile", daemon=True)
        t.start()
        _state["writer"] = t
        import atexit

        atexit.register(lambda: write_snapshot())
    try:
        write_snapshot()
    except Exception:
        pass
