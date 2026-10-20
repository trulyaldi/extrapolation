"""
What the add-in build registers in Excel: the xlwings metadata of every UDF in
excel/extrapolation.py.

On Linux xw.func/xw.arg/xw.ret are no-ops, so a subprocess stubs the Windows
modules, imports xlwings.udfs, installs its real decorators and then imports
extrapolation, the way excel/build/build_addin.py sees it on Windows.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

DUMP = """
    import json, sys
    from unittest.mock import MagicMock
    for name in ("pythoncom", "pywintypes", "win32api", "win32con", "win32com", "win32com.client",
                 "win32com.server", "win32com.server.util", "win32com.server.dispatcher",
                 "win32com.server.policy", "win32com.server.register"):
        sys.modules[name] = MagicMock()
    import xlwings as xw
    from xlwings import udfs
    xw.func, xw.arg, xw.ret = udfs.xlfunc, udfs.xlarg, udfs.xlret
    import extrapolation
    out = {}
    for name in extrapolation.__all__:
        f = getattr(extrapolation, name).__xlfunc__
        out[name] = {
            "volatile": f["volatile"],
            "category": f["category"],
            "ret": {k: v for k, v in f["ret"]["options"].items()},
            "args": [{"name": a["name"], "optional": "optional" in a, "doc": a["doc"],
                      "options": {k: repr(v) for k, v in a["options"].items()}} for a in f["args"]],
        }
    print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def registered():
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXTRAPLUS_")}
    env.update(PYTHONPATH=str(TESTS.parent / "excel"), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               EXTRAPLUS_CACHE="off")
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(DUMP)], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


MODELS = (1, 2, 3)
NUMERIC = ([f"{base}{n}" for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE", "FINITE_EXTRAP") for n in MODELS]
           + [f"EXTRAP{n}{s}" for s in ("_V", "_H") for n in MODELS])
PLOTS = [f"EXTRAP{n}_PLOT{s}" for s in ("", "_LOG") for n in MODELS]

# The arguments as Excel sees them in 0.3.0 (names, and whether VBA marks them
# Optional).  Existing functions must keep them.
SIGNATURES_0_3_0 = {
    **{f"{base}{n}": [("x_range", False), ("y_range", False), ("b_fixed", True), ("n_fit", True)]
       for base in ("EXTRAP", "UNCERTAINTY", "DECAY_RATE") for n in MODELS},
    **{f"EXTRAP{n}{s}": [("x_range", False), ("y_range", False), ("b_fixed", True), ("n_fit", True)]
       for s in ("_V", "_H") for n in MODELS},
    **{f"FINITE_EXTRAP{n}": [("x_range", False), ("y_range", False), ("target", True), ("b_fixed", True),
                             ("n_fit", True)] for n in MODELS},
    **{name: [("x_range", False), ("y_range", False), ("caller", False), ("b_fixed", True), ("n_fit", True)]
       for name in PLOTS},
    "EXTRAPLUS_VERSION": [],
}


def test_no_function_is_volatile(registered):
    assert [name for name, f in registered.items() if f["volatile"]] == []


def test_every_function_is_in_the_extrapolation_category(registered):
    assert {f["category"] for f in registered.values()} == {"Extrapolation"}


def test_existing_functions_keep_their_arguments(registered):
    for name, expected in SIGNATURES_0_3_0.items():
        assert [(a["name"], a["optional"]) for a in registered[name]["args"]] == expected, name


def test_only_the_existing_spill_functions_use_expand(registered):
    expanding = sorted(name for name, f in registered.items() if f["ret"].get("expand"))
    assert expanding == sorted(f"EXTRAP{n}{s}" for s in ("_V", "_H") for n in MODELS)


def test_excel_errors_in_b_fixed_and_n_fit_are_read_as_text(registered):
    for name in NUMERIC:
        options = {a["name"]: a["options"] for a in registered[name]["args"]}
        assert options["b_fixed"].get("err_to_str") == "True", name
        assert options["n_fit"].get("err_to_str") == "True", name


def test_plot_functions_are_unchanged(registered):
    for name in PLOTS:
        for arg in registered[name]["args"]:
            assert "err_to_str" not in arg["options"], name
        assert registered[name]["ret"] == {}, name
