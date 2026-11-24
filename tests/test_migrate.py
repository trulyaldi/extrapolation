"""excel/extraplus_migrate.py: only the old add-in's link prefix in front of Extraplus functions is removed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "excel"))

import extraplus_migrate as m  # noqa: E402

REAL = r"'C:\Users\me\AppData\Local\Extraplus\extraplus.xlam'!"


@pytest.mark.parametrize("old, new", [
    (f"={REAL}EXTRAP1_V($A$2:$A$6,B2:B6)", "=EXTRAP1_V($A$2:$A$6,B2:B6)"),
    (f"={REAL}EXTRAP2_V($AF3:$AF7,AH3:AH7,AJ5)", "=EXTRAP2_V($AF3:$AF7,AH3:AH7,AJ5)"),
    (f"={REAL}EXTRAP1(A1:A5,B1:B5)+{REAL}UNCERTAINTY1(A1:A5,B1:B5)",
     "=EXTRAP1(A1:A5,B1:B5)+UNCERTAINTY1(A1:A5,B1:B5)"),
    ("=extraplus.xlam!EXTRAP3_H(A1:A5,B1:B5)", "=EXTRAP3_H(A1:A5,B1:B5)"),
    ("=EXTRAPLUS.XLAM!finite_extrap1(A1:A5,B1:B5,1000)", "=finite_extrap1(A1:A5,B1:B5,1000)"),
    (r"='C:\Users\ALDIYA~1\AppData\Local\Extraplus\extraplus.xlam'!EXTRAP_ALL_V(A1:A5,B1:B5)",
     "=EXTRAP_ALL_V(A1:A5,B1:B5)"),
    (r"='C:\Users\O''Brien\Extraplus\extraplus.xlam'!DECAY_RATE2(A1:A5,B1:B5)", "=DECAY_RATE2(A1:A5,B1:B5)"),
    (r"='C:\Extraplus-u12ab\extraplus.xlam'!EXTRAP1_PLOT_LOG(A1:A5,B1:B5)", "=EXTRAP1_PLOT_LOG(A1:A5,B1:B5)"),
    (f"=IF(A1>0,{REAL}EXTRAP1(A1:A5,B1:B5),0)", "=IF(A1>0,EXTRAP1(A1:A5,B1:B5),0)"),
    (f"={REAL}EXTRAPLUS_VERSION()", "=EXTRAPLUS_VERSION()"),
])
def test_prefix_is_removed(old, new):
    assert m.convert_formula(old) == new


@pytest.mark.parametrize("text", [
    "=EXTRAP1_V(A1:A5,B1:B5)",                        # already converted
    "=SUM(A1:A5)",
    "=[Book2.xlsx]Sheet1!A1",
    r"='C:\data\[extraplus.xlam.xlsx]Sheet1'!A1",     # a cell of another workbook
    f"={REAL}MY_OWN_FUNCTION(A1)",                    # not one of ours: left alone
    "extraplus.xlam!EXTRAP1(A1:A5,B1:B5)",            # text, not a formula... still only the prefix
    None, 3.0,
])
def test_other_formulas_are_left_alone(text):
    out = m.convert_formula(text)
    if isinstance(text, str) and text.startswith("extraplus.xlam!"):
        assert out == "EXTRAP1(A1:A5,B1:B5)"
    else:
        assert out is None


def test_prefixes_for_replace():
    text = f"={REAL}EXTRAP1(A1:A5,B1:B5)+{REAL}EXTRAP2(A1:A5,B1:B5)"
    assert m.prefixes_in(text) == {REAL}


def test_every_function_is_known():
    import extraplus_xlcompat as c
    assert set(m.FUNCTION_NAMES) == set(c.SIGNATURES)
