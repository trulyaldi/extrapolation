"""
excel/extraplus_profile.py (performance investigation): off by default, and when
on it must not change any UDF output, name or argument.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
EXCEL = TESTS.parent / "excel"

CHILD = """
    import json, sys, extrapolation
    from udf_cases import udf_outputs, FAILING
    out = udf_outputs(extrapolation)
    out.update({"failing/" + k: v for k, v in udf_outputs(extrapolation, FAILING).items()})
    names = {n: getattr(extrapolation, n).__name__ for n in extrapolation.__all__}
    print(json.dumps({"outputs": {k: repr(v) for k, v in out.items()}, "names": names,
                      "profiled": "extraplus_profile" in sys.modules,
                      "wrapped": getattr(extrapolation._run_udf, "_extraplus_profiled", False)}))
"""


def run(profile, tmp_path, cache="off"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXTRAPLUS_")}
    env.update(PYTHONPATH=os.pathsep.join([str(TESTS), str(EXCEL)]), PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", EXTRAPLUS_CACHE=cache, EXTRAPLUS_PROFILE_DIR=str(tmp_path))
    if profile is not None:
        env["EXTRAPLUS_PROFILE"] = profile
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def unprofiled(tmp_path_factory):
    return run(None, tmp_path_factory.mktemp("off"))


def test_off_by_default_nothing_is_wrapped(unprofiled):
    assert not unprofiled["profiled"]
    assert not unprofiled["wrapped"]


@pytest.mark.parametrize("level", ["1", "2"])
@pytest.mark.parametrize("cache", ["off", "memory"])
def test_outputs_and_names_identical_when_profiled(level, cache, unprofiled, tmp_path):
    on = run(level, tmp_path, cache=cache)
    assert on["profiled"] and on["wrapped"]
    assert on["names"] == unprofiled["names"]
    assert on["outputs"] == unprofiled["outputs"]
    snap = json.loads(next(tmp_path.glob("extraplus-profile-*.json")).read_text())
    assert snap["counts"]["ours._run_udf"] > 0


def test_snapshot_counts_b_fixed_kinds(tmp_path):
    code = """
        import json, math, extrapolation, extraplus_profile
        from udf_cases import column
        X, Y = column([100, 200, 300, 400, 500]), column([-1.0, -1.5, -1.7, -1.8, -1.85])
        for b in (None, float("nan"), "#NUM!", 2.0):
            extrapolation.EXTRAP1_V(X, Y, b)
        print(json.dumps(extraplus_profile.snapshot()))
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXTRAPLUS_")}
    env.update(PYTHONPATH=os.pathsep.join([str(TESTS), str(EXCEL)]), MPLBACKEND="Agg",
               EXTRAPLUS_CACHE="off", EXTRAPLUS_PROFILE="1", EXTRAPLUS_PROFILE_DIR=str(tmp_path))
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    snap = json.loads(done.stdout.strip().splitlines()[-1])
    assert snap["b_fixed_kind"] == {"none": 1, "nan": 1, "error_text": 1, "number": 1}
