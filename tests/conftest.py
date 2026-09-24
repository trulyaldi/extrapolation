"""Shared test data. Files are located relative to this file, never the working directory."""

import matplotlib

matplotlib.use("Agg")   # before anything imports pyplot

import pytest  # noqa: E402

# Be 1Se ground-state energy, first 12 basis sizes of
# research/data/large-dataset/Be_1Se.csv (the converged value is -14.6673565...).
BE_1SE_X = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
BE_1SE_Y = [
    -14.6661876127332, -14.6667402339119, -14.6672828224583, -14.6673249180953,
    -14.66734057696, -14.6673445259951, -14.6673478029866, -14.6673503395393,
    -14.6673519816691, -14.6673531093866, -14.6673536576132, -14.6673541629258,
]


@pytest.fixture
def be_xy():
    return list(BE_1SE_X), list(BE_1SE_Y)
