"""
The "write values" command (extrapolation._values_pass) on a fake workbook.

The fake stands in for _XlwingsBook: cells, formulas and array formulas in
dicts, with the two xlwings read conversions the UDF arguments get (np.array
with ndim=2, blank cells as NaN; raw values, blank cells as None, Excel errors
as text with err_to_str).  The real Excel side is checked by hand (see the
manual test checklist).
"""

import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "excel"))

import extrapolation  # noqa: E402
from conftest import BE_1SE_X, BE_1SE_Y  # noqa: E402

N = len(BE_1SE_X)
PROP_Y = [v * 1.5 + 0.1 for v in BE_1SE_Y]


class FakeBook:
    def __init__(self, sheet="2P"):
        self.sheet = sheet
        self.cells = {}        # (sheet, row, col) -> value
        self.formulas = {}     # (sheet, row, col) -> formula text
        self.arrays = {}       # (sheet, row, col) of each cell -> box of its array formula
        self.backups = 0
        self.recorded = None
        self.paused = []
        self.fail_write_at = None
        self.writes = []

    # building the sheet
    def column(self, col, values, row=2):
        for i, v in enumerate(values):
            self.cells[(self.sheet, row + i, col)] = v

    def formula(self, row, col, text, rows=1, cols=1, shown="#NAME?"):
        box = (row, col, row + rows - 1, col + cols - 1)
        for r in range(row, row + rows):
            for c in range(col, col + cols):
                self.formulas[(self.sheet, r, c)] = text
                self.cells[(self.sheet, r, c)] = shown
                if rows * cols > 1:
                    self.arrays[(self.sheet, r, c)] = box

    # the adapter interface
    def formula_cells(self):
        for (sheet, row, col), text in sorted(self.formulas.items(), key=lambda kv: (kv[0][2], kv[0][1])):
            if extrapolation._MENTIONS_RE.search(text):
                yield sheet, row, col, text

    def block_of(self, sheet, row, col):
        return self.arrays.get((sheet, row, col), (row, col, row, col))

    def address(self, sheet, box):
        return f"{sheet}!R{box[0]}C{box[1]}:R{box[2]}C{box[3]}"

    def _grid(self, sheet, box):
        return [[self.cells.get((sheet, r, c)) for c in range(box[1], box[3] + 1)]
                for r in range(box[0], box[2] + 1)]

    def read(self, sheet, area, options):
        grid = self._grid(sheet, extrapolation._area_box(area))
        return self.convert(grid, options)

    def convert(self, value, options):
        def clean(v):
            if isinstance(v, str) and v.startswith("#"):
                return v if options.get("err_to_str") else None
            return v
        grid = value if isinstance(value, list) else [[value]]
        grid = [[clean(v) for v in row] for row in grid]
        if options.get("convert") is np.array:   # NumpyArrayConverter: empty=np.nan, ndmin=ndim
            return np.array([[np.nan if v in (None, "") else v for v in row] for row in grid],
                            ndmin=options.get("ndim") or 0)
        if options.get("ndim") == 2:
            return grid
        return grid[0][0] if len(grid) == 1 and len(grid[0]) == 1 else grid

    def evaluate(self, text):
        return float(text)

    def values(self, sheet, box):
        return self._grid(sheet, box)

    def write(self, sheet, box, result):
        if self.fail_write_at is not None and len(self.writes) == self.fail_write_at:
            raise RuntimeError("Excel refused the write")
        out = np.asarray(result, dtype=object).reshape(box[2] - box[0] + 1, box[3] - box[1] + 1)
        for i, r in enumerate(range(box[0], box[2] + 1)):
            for j, c in enumerate(range(box[1], box[3] + 1)):
                v = out[i, j]
                self.cells[(sheet, r, c)] = None if isinstance(v, float) and np.isnan(v) else v
                self.formulas.pop((sheet, r, c), None)
                self.arrays.pop((sheet, r, c), None)
        self.writes.append(box)

    def backup(self):
        self.backups += 1
        return "C:\\work\\book (before Extraplus values).xlsm"

    @contextmanager
    def calculation_paused(self):
        self.paused.append("enter")
        try:
            yield
        finally:
            self.paused.append("exit")

    def record(self, records, report):
        self.recorded = records


def energy_sheet():
    """x in A2:A13, energy in B2:B13, property in C2:C13."""
    book = FakeBook()
    book.column(1, BE_1SE_X)
    book.column(2, BE_1SE_Y)
    book.column(3, PROP_Y)
    return book


def col(values):
    return np.asarray(values, dtype=object).reshape(-1, 1)


def bits(values):
    return [struct.pack("<d", float(v)) for v in np.asarray(values, dtype=object).ravel()]


def written(book, box):
    return [book.cells[(book.sheet, r, c)] for r in range(box[0], box[2] + 1) for c in range(box[1], box[3] + 1)]


def test_array_and_single_cell_formulas_become_their_values():
    book = energy_sheet()
    book.formula(2, 5, "=EXTRAP1_V($A$2:$A$13,B2:B13)", rows=3)
    book.formula(2, 6, "=EXTRAP2(A2:A13,B2:B13)")
    book.formula(2, 7, "=FINITE_EXTRAP3(A2:A13,B2:B13,15000)")
    report = extrapolation._values_pass(book)
    assert (report["written"], report["failed"], report["skipped"]) == (3, 0, 0)
    assert bits(written(book, (2, 5, 4, 5))) == bits(extrapolation.EXTRAP1_V(col(BE_1SE_X), col(BE_1SE_Y)))
    assert bits(written(book, (2, 6, 2, 6))) == bits([extrapolation.EXTRAP2(col(BE_1SE_X), col(BE_1SE_Y))])
    assert bits(written(book, (2, 7, 2, 7))) == bits(
        [extrapolation.FINITE_EXTRAP3(col(BE_1SE_X), col(BE_1SE_Y), 15000.0)])
    assert not book.formulas
    assert book.backups == 1 and book.paused == ["enter", "exit"]
    assert [r[1] for r in book.recorded] == ["=EXTRAP1_V($A$2:$A$13,B2:B13)", "=EXTRAP2(A2:A13,B2:B13)",
                                              "=FINITE_EXTRAP3(A2:A13,B2:B13,15000)"]
    assert report["changed"] == 3   # they showed #NAME?


def test_chained_b_is_computed_first_and_read_from_the_written_cells():
    book = energy_sheet()
    # The property block (column E) comes before the energy block (column G) on the sheet.
    book.formula(2, 5, "=EXTRAP1_V(A2:A13,C2:C13,G4)", rows=3)
    book.formula(2, 7, "=EXTRAP1_V(A2:A13,B2:B13)", rows=3)
    report = extrapolation._values_pass(book)
    assert report["written"] == 2
    energy = extrapolation.EXTRAP1_V(col(BE_1SE_X), col(BE_1SE_Y))
    prop = extrapolation.EXTRAP1_V(col(BE_1SE_X), col(PROP_Y), float(energy[2, 0]))
    assert bits(written(book, (2, 7, 4, 7))) == bits(energy)
    assert bits(written(book, (2, 5, 4, 5))) == bits(prop)
    assert book.writes == [(2, 7, 4, 7), (2, 5, 4, 5)]


def test_when_the_energy_fails_the_dependent_formula_is_kept():
    book = energy_sheet()
    book.formula(2, 7, "=EXTRAP1_V(A2:A3,B2:B3)", rows=3)          # two points: an error
    book.formula(2, 5, "=EXTRAP1_V(A2:A13,C2:C13,G4)", rows=3)
    report = extrapolation._values_pass(book)
    assert (report["written"], report["failed"], report["skipped"]) == (0, 1, 1)
    assert len(book.formulas) == 6 and book.backups == 0 and book.recorded is None
    reasons = dict((where, (status, why)) for where, status, why in report["problems"])
    assert reasons["2P!R2C7:R4C7"][0] == "failed" and "three paired" in reasons["2P!R2C7:R4C7"][1]
    assert reasons["2P!R2C5:R4C5"] == ("skipped", "uses 2P!R2C7:R4C7, which was not written")


def test_extrap_all_v_with_the_b_row_of_another():
    book = energy_sheet()
    book.formula(2, 5, "=EXTRAP_ALL_V(A2:A13,B2:B13)", rows=3, cols=3)
    book.formula(6, 5, "=EXTRAP_ALL_V(A2:A13,C2:C13,E4:G4)", rows=3, cols=3)
    report = extrapolation._values_pass(book)
    assert report["written"] == 2
    energy = extrapolation.EXTRAP_ALL_V(col(BE_1SE_X), col(BE_1SE_Y))
    prop = extrapolation.EXTRAP_ALL_V(col(BE_1SE_X), col(PROP_Y), energy[2:3].tolist())
    assert bits(written(book, (6, 5, 8, 7))) == bits(prop)


def test_unresolved_add_in_path_and_literal_arguments():
    book = energy_sheet()
    book.formula(2, 5, "='C:\\Users\\User\\AppData\\Local\\Extraplus\\extraplus.xlam'!EXTRAP3_V(A2:A13,B2:B13)",
                 rows=3)
    book.formula(2, 6, "=EXTRAP1(A2:A13,B2:B13,5)")
    book.formula(2, 7, "=UNCERTAINTY2(A2:A13,B2:B13,,6)")
    report = extrapolation._values_pass(book)
    assert report["written"] == 3
    assert bits(written(book, (2, 6, 2, 6))) == bits([extrapolation.EXTRAP1(col(BE_1SE_X), col(BE_1SE_Y), 5.0)])
    assert bits(written(book, (2, 7, 2, 7))) == bits(
        [extrapolation.UNCERTAINTY2(col(BE_1SE_X), col(BE_1SE_Y), None, 6.0)])


@pytest.mark.parametrize("text", [
    "=INDEX(EXTRAP1_V(A2:A13,B2:B13),1)",      # nested
    "=EXTRAP1(xs,ys)",                         # names
    "=EXTRAP1([Other.xlsx]Sheet1!A2:A13,B2:B13)",
    '=EXTRAP1(A2:A13,B2:B13,"5")',
    "=EXTRAP1(A2:A13)",                        # too few arguments
    "=EXTRAP1(A2:A13,B2:B13,1,2,3)",           # too many
])
def test_other_formulas_are_left_and_listed(text):
    book = energy_sheet()
    book.formula(2, 5, text)
    report = extrapolation._values_pass(book)
    assert report["written"] == 0 and report["unrecognized"] == 1
    assert book.formulas == {("2P", 2, 5): text}


def test_formulas_that_are_not_extraplus_are_not_touched_or_listed():
    book = energy_sheet()
    book.formula(2, 5, "=SUM(B2:B13)")
    book.formula(2, 6, "=EXTRAP1_PLOT(A2:A13,B2:B13)")
    assert extrapolation._values_pass(book)["problems"] == []
    assert len(book.formulas) == 2


def test_a_result_that_does_not_fit_the_formulas_cells_is_kept():
    book = energy_sheet()
    book.formula(2, 5, "=EXTRAP1_V(A2:A13,B2:B13)")   # one cell, not resized to 3
    report = extrapolation._values_pass(book)
    assert report["skipped"] == 1 and report["written"] == 0
    assert "3 x 1" in report["problems"][0][2]


def test_excel_error_in_b_cell_is_an_error_not_a_free_fit():
    book = energy_sheet()
    book.cells[("2P", 2, 9)] = "#N/A"
    book.formula(2, 5, "=EXTRAP1_V(A2:A13,B2:B13,I2)", rows=3)
    report = extrapolation._values_pass(book)
    assert report["failed"] == 1 and "b_fixed must be a number" in report["problems"][0][2]


def test_a_write_error_stops_and_keeps_what_was_written():
    book = energy_sheet()
    book.formula(2, 5, "=EXTRAP1(A2:A13,B2:B13)")
    book.formula(2, 6, "=EXTRAP2(A2:A13,B2:B13)")
    book.fail_write_at = 1
    report = extrapolation._values_pass(book)
    assert report["stopped"] == "Error: RuntimeError: Excel refused the write"
    assert report["written"] == 1 and len(book.recorded) == 1
    assert book.paused == ["enter", "exit"]
    assert "Stopped early" in extrapolation._summary(report)


@pytest.mark.parametrize("text, expected", [
    ("=EXTRAP1($A$2:$A$13,'My ''data'' sheet'!B2:B13)",
     ("EXTRAP1", [("ref", "2P", "$A$2:$A$13", (2, 1, 13, 1)), ("ref", "My 'data' sheet", "B2:B13", (2, 2, 13, 2))])),
    ("=extrap2_v(A:A,Sheet2!B:B)",
     ("EXTRAP2_V", [("ref", "2P", "A:A", (1, 1, 1_048_576, 1)), ("ref", "Sheet2", "B:B", (1, 2, 1_048_576, 2))])),
    ("=EXTRAP1(A2:A13, B2:B13, , 1E-3)",
     ("EXTRAP1", [("ref", "2P", "A2:A13", (2, 1, 13, 1)), ("ref", "2P", "B2:B13", (2, 2, 13, 2)), ("missing",),
                  ("number", "1E-3")])),
    ("=extraplus.xlam!DECAY_RATE3(AA10:AB10,AC10:AD10)",
     ("DECAY_RATE3", [("ref", "2P", "AA10:AB10", (10, 27, 10, 28)), ("ref", "2P", "AC10:AD10", (10, 29, 10, 30))])),
    ("=EXTRAPLUS_VERSION()", None),
    ("=EXTRAP1_PLOT(A2:A13,B2:B13)", None),
    ("=EXTRAP1(A2:A13,B2:B13)+1", None),
])
def test_parse(text, expected):
    assert extrapolation._parse_call(text, "2P") == expected
