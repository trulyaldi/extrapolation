"""Compare Excel results from e2e_excel.py with the same inputs computed in
another environment (for example the research .venv in WSL):

    .venv/bin/python excel/build/e2e_compare.py results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_reference import reference  # noqa: E402


def main(path):
    results = json.loads(Path(path).read_text())
    x, y = zip(*results["inputs"])
    worst = 0.0
    print(f"{'function':14s} {'excel':>24s} {'this env':>24s} {'rel diff':>10s}")
    for row in results["comparison"]["rows"]:
        n = int(row["function"][-1])
        kind = row["function"][:-1]
        mine = reference(x, y, n)[{"EXTRAP": 0, "UNCERTAINTY": 1, "DECAY_RATE": 2}[kind]]
        rel = abs(row["excel"] - mine) / abs(mine) if mine else abs(row["excel"] - mine)
        worst = max(worst, rel)
        print(f"{row['function']:14s} {row['excel']:24.16g} {mine:24.16g} {rel:10.2e}")
    print(f"max relative difference: {worst:.3e}")


if __name__ == "__main__":
    main(sys.argv[1])
