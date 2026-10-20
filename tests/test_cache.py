"""
The result cache in excel/extrapolation.py: exact keys, bit-identical outputs
with the cache off, cold, warm and reloaded from the file (also in a new
process), failures never cached, and a damaged, locked or unwritable file
falling back to computing.
"""

import json
import math
import os
import sqlite3
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "excel"))

import extrapolation  # noqa: E402
from udf_cases import DATASETS, FAILING, column, udf_outputs  # noqa: E402


def make_cache(monkeypatch, folder=None, mode=None, **extra_env):
    """Install a fresh cache as the one the UDFs use; return it."""
    env = dict(extra_env)
    if folder is not None:
        env["EXTRAPLUS_CACHE_DIR"] = str(folder)
    if mode is not None:
        env["EXTRAPLUS_CACHE"] = mode
    cache = extrapolation._ResultCache(env)
    monkeypatch.setattr(extrapolation, "_CACHE", cache)
    return cache


def db_rows(folder):
    con = sqlite3.connect(str(Path(folder) / extrapolation._DISK_FILE))
    try:
        return con.execute("SELECT count(*) FROM results").fetchone()[0]
    finally:
        con.close()


def run_child(code, folder, **env):
    """Run code in a new Python with the cache file in folder; return its stdout as JSON."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("EXTRAPLUS_")}
    child_env.update(PYTHONPATH=os.pathsep.join([str(TESTS), str(TESTS.parent / "excel")]),
                     PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                     EXTRAPLUS_CACHE_DIR=str(folder), **env)
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=child_env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


CHILD_OUTPUTS = """
    import json, extrapolation
    from udf_cases import udf_outputs
    out = udf_outputs(extrapolation)
    c = extrapolation._CACHE
    print(json.dumps({"outputs": {k: repr(v) for k, v in out.items()},
                      "hits_disk": c.hits_disk, "misses": c.misses, "state": c.disk.state}))
"""


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def key(*cells, model="exponential", b_fixed=None, n_fit=None):
    cache = extrapolation._ResultCache({})
    x, y = cells
    return cache._key(tuple(x), tuple(y), model, b_fixed, n_fit)


def test_key_is_stable_across_cache_instances():
    assert key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])


@pytest.mark.parametrize("a, b", [
    (0.0, -0.0),
    (1.0, 1),
    (1, True),
    (1.0, True),
    (1.0, "1"),
    (None, ""),
    (float("nan"), struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]),   # NaN payloads
    (float("inf"), 1.7976931348623157e308),
    (0.1, 0.1 + 2 ** -56),          # one ulp apart
    (0.30000000000000004, 0.3),
])
def test_key_distinguishes_exact_cell_values(a, b):
    assert key([1.0, 2.0, a], [3.0, 2.0, 1.0]) != key([1.0, 2.0, b], [3.0, 2.0, 1.0])
    assert key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], b_fixed=a) != key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], b_fixed=b)


def test_key_distinguishes_model_and_where_x_ends():
    assert key([1.0, 2.0], [3.0]) != key([1.0], [2.0, 3.0])
    assert key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) != key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], model="power_law")
    assert key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], n_fit=3) != key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], n_fit=3.0)


def test_key_depends_on_the_code_fingerprint(monkeypatch):
    before = key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    monkeypatch.setattr(extrapolation.extraplus, "__version__", "9.9.9")
    assert key([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) != before


def test_values_without_an_exact_encoding_are_computed_not_cached(monkeypatch, be_xy):
    import datetime
    cache = make_cache(monkeypatch)
    x, y = be_xy
    assert isinstance(extrapolation.EXTRAP1(column(x), column(y), [[1.0, 2.0]]), str)   # range as b_fixed
    padded = column([*x, datetime.datetime(2026, 1, 1)])
    assert extrapolation.EXTRAP1(padded, column([*y, 1.0])) == extrapolation.EXTRAP1(column(x), column(y))
    assert cache.uncacheable == 2 and len(cache._memory) == 1


def test_result_encoding_round_trips_exactly(be_xy):
    from extraplus import extrapolate
    result = extrapolate(*be_xy, "power_law", 2.5, 7)
    decoded = extrapolation._decode_result(extrapolation._encode_result(result))
    assert decoded == result
    for name in extrapolation._RESULT_FIELDS:
        assert repr(getattr(decoded, name)) == repr(getattr(result, name))


def test_results_are_deterministic(be_xy):
    """The assumption the cache rests on: the same input gives the same bits."""
    from udf_cases import bits
    from extraplus import extrapolate
    for model in ("exponential", "sqrt_exponential", "power_law"):
        first, second = extrapolate(*be_xy, model), extrapolate(*be_xy, model)
        for name in extrapolation._RESULT_FIELDS:
            assert bits(getattr(first, name)) == bits(getattr(second, name))


# ---------------------------------------------------------------------------
# Bit-identical outputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_outputs():
    """Every output with the cache off."""
    mp = pytest.MonkeyPatch()
    try:
        make_cache(mp, mode="off")
        return udf_outputs(extrapolation)
    finally:
        mp.undo()


def test_outputs_identical_off_cold_warm_and_from_the_file(monkeypatch, tmp_path, reference_outputs):
    cold_cache = make_cache(monkeypatch, tmp_path)
    cold = udf_outputs(extrapolation)
    assert cold_cache.misses == 3 * len(DATASETS) and cold_cache.hits_disk == 0
    warm = udf_outputs(extrapolation)
    assert cold_cache.misses == 3 * len(DATASETS)

    file_cache = make_cache(monkeypatch, tmp_path)   # empty memory, same file
    from_file = udf_outputs(extrapolation)
    assert file_cache.misses == 0 and file_cache.hits_disk == 3 * len(DATASETS)

    memory_cache = make_cache(monkeypatch, mode="memory")
    memory_only = udf_outputs(extrapolation)
    assert memory_cache.disk is None

    for name, outputs in (("cold", cold), ("warm", warm), ("file", from_file), ("memory", memory_only)):
        assert outputs == reference_outputs, name


def test_outputs_identical_when_read_back_by_a_new_process(tmp_path, reference_outputs):
    first = run_child(CHILD_OUTPUTS, tmp_path)
    second = run_child(CHILD_OUTPUTS, tmp_path)
    assert first["misses"] == 3 * len(DATASETS) and first["hits_disk"] == 0
    assert second["misses"] == 0 and second["hits_disk"] == 3 * len(DATASETS)
    expected = {k: repr(v) for k, v in reference_outputs.items()}
    assert first["outputs"] == expected
    assert second["outputs"] == expected


def test_nan_result_comes_back_with_the_same_bits(monkeypatch, tmp_path):
    x, y, *_ = DATASETS["nan_result"]
    make_cache(monkeypatch, tmp_path)
    fresh = extrapolation._run_udf(column(x), column(y), "exponential")
    make_cache(monkeypatch, tmp_path)
    again = extrapolation._run_udf(column(x), column(y), "exponential")
    assert math.isnan(fresh.value)
    assert struct.pack("<d", again.value) == struct.pack("<d", fresh.value)


# ---------------------------------------------------------------------------
# What is not cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAILING)
def test_input_errors_are_never_cached(name, monkeypatch, tmp_path):
    x, y, b_fixed, n_fit = FAILING[name]
    cache = make_cache(monkeypatch, tmp_path)
    for _ in range(2):
        result = extrapolation.EXTRAP1(column(x), column(y), b_fixed, n_fit)
        assert isinstance(result, str) and result.startswith("Error: ")
    assert cache.misses == 2 and not cache._memory
    assert db_rows(tmp_path) == 0


def test_engine_failure_is_not_cached(monkeypatch, tmp_path, be_xy):
    """An environment-dependent failure (here simulated) must not stick."""
    cache = make_cache(monkeypatch, tmp_path)
    real = extrapolation._extrapolate_quietly

    def broken(*args):
        raise MemoryError("simulated")

    monkeypatch.setattr(extrapolation, "_extrapolate_quietly", broken)
    X, Y = column(be_xy[0]), column(be_xy[1])
    assert extrapolation.EXTRAP1(X, Y) == "Error: MemoryError: simulated"
    monkeypatch.setattr(extrapolation, "_extrapolate_quietly", real)
    assert isinstance(extrapolation.EXTRAP1(X, Y), float)
    assert cache.misses == 2 and db_rows(tmp_path) == 1


# ---------------------------------------------------------------------------
# A damaged, locked or unwritable file
# ---------------------------------------------------------------------------


def expected_value(be_xy):
    from extraplus import extrapolate
    return extrapolate(*be_xy).value


def test_garbage_file_is_set_aside_and_replaced(monkeypatch, tmp_path, be_xy):
    (tmp_path / extrapolation._DISK_FILE).write_bytes(b"this is not an sqlite file" * 100)
    make_cache(monkeypatch, tmp_path)
    X, Y = column(be_xy[0]), column(be_xy[1])
    assert extrapolation.EXTRAP1(X, Y) == expected_value(be_xy)
    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1
    make_cache(monkeypatch, tmp_path)   # the next process uses the new file
    assert extrapolation.EXTRAP1(X, Y) == expected_value(be_xy)
    assert db_rows(tmp_path) == 1


def test_truncated_file_falls_back_to_computing(monkeypatch, tmp_path, reference_outputs):
    make_cache(monkeypatch, tmp_path)
    udf_outputs(extrapolation)
    extrapolation._CACHE.disk.close()
    path = tmp_path / extrapolation._DISK_FILE
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    for side in tmp_path.glob("*-wal"):
        side.unlink()
    cache = make_cache(monkeypatch, tmp_path)
    assert udf_outputs(extrapolation) == reference_outputs
    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1
    assert cache.disk.state == "open" and db_rows(tmp_path) == 3 * len(DATASETS)


def test_damaged_row_is_recomputed_and_dropped(monkeypatch, tmp_path, be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    make_cache(monkeypatch, tmp_path)
    extrapolation.EXTRAP1(X, Y)
    extrapolation._CACHE.disk.close()
    con = sqlite3.connect(str(tmp_path / extrapolation._DISK_FILE))
    con.execute("UPDATE results SET value = substr(value, 1, length(value) - 1) || x'00'")
    con.commit()
    con.close()
    cache = make_cache(monkeypatch, tmp_path)
    assert extrapolation.EXTRAP1(X, Y) == expected_value(be_xy)
    assert cache.misses == 1 and cache.hits_disk == 0


def test_locked_file_does_not_block_or_fail(monkeypatch, tmp_path, be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    make_cache(monkeypatch, tmp_path)
    extrapolation.EXTRAP1(X, Y)            # creates the file
    other = sqlite3.connect(str(tmp_path / extrapolation._DISK_FILE), isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")        # another Excel holding the write lock
    try:
        cache = make_cache(monkeypatch, tmp_path)
        for model in (2, 3):
            value = getattr(extrapolation, f"EXTRAP{model}")(X, Y)
            assert isinstance(value, float)
        assert cache.misses == 2
        assert not cache.disk.state.startswith(("disabled", "error"))
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert isinstance(extrapolation.EXTRAP3(column(be_xy[0][:6]), column(be_xy[1][:6])), float)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
def test_unwritable_folder_means_memory_only(monkeypatch, tmp_path, be_xy):
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        cache = make_cache(monkeypatch, folder)
        X, Y = column(be_xy[0]), column(be_xy[1])
        assert extrapolation.EXTRAP1(X, Y) == expected_value(be_xy)
        assert extrapolation.EXTRAP1(X, Y) == expected_value(be_xy)
        assert cache.hits_memory == 1
        assert cache.disk.state.startswith("error")
    finally:
        folder.chmod(0o700)


def test_two_processes_write_the_same_file_at_once(tmp_path):
    code = """
        import json, sys, extrapolation
        from udf_cases import column, BE_1SE_X, BE_1SE_Y
        offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        for i in range(60):
            y = [v + (i + offset) * 1e-9 for v in BE_1SE_Y]
            assert isinstance(extrapolation.EXTRAP1(column(BE_1SE_X), column(y)), float)
        c = extrapolation._CACHE
        print(json.dumps({"misses": c.misses, "state": c.disk.state}))
    """
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("EXTRAPLUS_")}
    child_env.update(PYTHONPATH=os.pathsep.join([str(TESTS), str(TESTS.parent / "excel")]),
                     PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", EXTRAPLUS_CACHE_DIR=str(tmp_path))
    procs = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), str(offset)], env=child_env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for offset in (0, 30)]   # 30 of the 60 inputs overlap
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=600)
        assert proc.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))
    assert all(r["state"] == "open" for r in results), results
    assert db_rows(tmp_path) == 90


# ---------------------------------------------------------------------------
# Limits and switches
# ---------------------------------------------------------------------------


def test_memory_is_capped_lru(monkeypatch):
    monkeypatch.setattr(extrapolation, "_MEMORY_ENTRIES", 3)
    cache = make_cache(monkeypatch, mode="memory")
    x = column([1.0, 2.0, 3.0, 4.0])
    ys = [column([3.0, 2.0, 1.5, 1.2 + i * 0.01]) for i in range(5)]
    for y in ys[:3]:
        extrapolation.EXTRAP1(x, y)
    extrapolation.EXTRAP1(x, ys[0])           # ys[0] is now the most recent
    extrapolation.EXTRAP1(x, ys[3])           # evicts ys[1]
    assert len(cache._memory) == 3
    misses = cache.misses
    extrapolation.EXTRAP1(x, ys[0])
    assert cache.misses == misses
    extrapolation.EXTRAP1(x, ys[1])
    assert cache.misses == misses + 1


def test_file_is_capped_oldest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(extrapolation, "_DISK_ENTRIES", 20)
    monkeypatch.setattr(extrapolation, "_DISK_EVICT_EVERY", 5)
    make_cache(monkeypatch, tmp_path)
    x = column([1.0, 2.0, 3.0, 4.0])
    for i in range(40):
        extrapolation.EXTRAP1(x, column([3.0, 2.0, 1.5, 1.2 + i * 0.001]))
    assert db_rows(tmp_path) <= 20
    cache = make_cache(monkeypatch, tmp_path)
    extrapolation.EXTRAP1(x, column([3.0, 2.0, 1.5, 1.2 + 39 * 0.001]))   # newest is kept
    extrapolation.EXTRAP1(x, column([3.0, 2.0, 1.5, 1.2]))                # oldest is gone
    assert (cache.hits_disk, cache.misses) == (1, 1)


def test_rows_of_other_versions_are_purged(monkeypatch, tmp_path, be_xy):
    X, Y = column(be_xy[0]), column(be_xy[1])
    make_cache(monkeypatch, tmp_path)
    extrapolation.EXTRAP1(X, Y)
    extrapolation._CACHE.disk.close()
    monkeypatch.setattr(extrapolation.extraplus, "__version__", "9.9.9")
    cache = make_cache(monkeypatch, tmp_path)
    extrapolation.EXTRAP2(X, Y)
    assert cache.misses == 1 and db_rows(tmp_path) == 1


@pytest.mark.parametrize("value, mode", [("off", "off"), ("0", "off"), ("FALSE", "off"), ("no", "off"),
                                         ("memory", "memory"), ("on", "on"), ("", "on")])
def test_switch_values(value, mode):
    assert extrapolation._ResultCache({"EXTRAPLUS_CACHE": value}).mode == mode


def test_cache_off_writes_nothing(monkeypatch, tmp_path, be_xy):
    cache = make_cache(monkeypatch, tmp_path, mode="off")
    extrapolation.EXTRAP1(column(be_xy[0]), column(be_xy[1]))
    assert not cache._memory and not list(tmp_path.iterdir())
    assert extrapolation.EXTRAPLUS_CACHE_STATS() == "cache off (EXTRAPLUS_CACHE=off)"


def test_default_file_is_under_the_install_folder(monkeypatch, tmp_path, be_xy):
    cache = make_cache(monkeypatch, EXTRAPLUS_HOME=str(tmp_path))
    extrapolation.EXTRAP1(column(be_xy[0]), column(be_xy[1]))
    assert cache.disk.path == tmp_path / "resultcache" / extrapolation._DISK_FILE
    assert cache.disk.path.exists()
    stats = extrapolation.EXTRAPLUS_CACHE_STATS()
    assert stats.startswith("cache on; memory 1 results; file ") and "(1 results, open)" in stats


def test_without_a_folder_the_cache_is_memory_only(monkeypatch, be_xy):
    cache = make_cache(monkeypatch)
    extrapolation.EXTRAP1(column(be_xy[0]), column(be_xy[1]))
    assert cache.disk is None and len(cache._memory) == 1
