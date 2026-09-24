"""Reference values for the end-to-end test: the extraplus Python API called
directly (extraplus.extrapolate() with its defaults), which the Excel functions
must reproduce.  Deliberately independent of excel/extrapolation.py."""

from extraplus import extrapolate

MODELS = {1: "exponential", 2: "sqrt_exponential", 3: "power_law"}

# Be 1Se energy, first 12 basis sizes of research/data/large-dataset/Be_1Se.csv.
DATA = [
    (100, -14.6661876127332),
    (200, -14.6667402339119),
    (300, -14.6672828224583),
    (400, -14.6673249180953),
    (500, -14.66734057696),
    (600, -14.6673445259951),
    (700, -14.6673478029866),
    (800, -14.6673503395393),
    (900, -14.6673519816691),
    (1000, -14.6673531093866),
    (1100, -14.6673536576132),
    (1200, -14.6673541629258),
]


def reference(x, y, n):
    """Return (C, uncertainty, B) for model n in the original y units."""
    result = extrapolate(x, y, model=MODELS[n])
    return (result.value, result.uncertainty, result.decay_rate)
