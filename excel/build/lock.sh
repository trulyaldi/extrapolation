#!/usr/bin/env bash
# Regenerate excel/requirements-windows.txt: the pinned, hashed dependencies
# (numpy, scipy, pandas, matplotlib, xlwings, ...) that install.ps1 installs.
# Runs in WSL/Linux. Needs uv (set UV=/path/to/uv if it is not on PATH).
set -euo pipefail
cd "$(dirname "$0")/../.."
UV="${UV:-uv}"
PYTHON_VERSION=$(sed -n "s/^ *PythonVersion *= *'\(.*\)'/\1/p" excel/pins.psd1)

"$UV" pip compile --quiet pyproject.toml \
    --extra excel \
    --constraints excel/build/constraints.txt \
    --python-platform x86_64-pc-windows-msvc \
    --python-version "$PYTHON_VERSION" \
    --only-binary :all: \
    --generate-hashes \
    --no-header \
    --output-file excel/requirements-windows.txt

echo "Wrote excel/requirements-windows.txt for Python $PYTHON_VERSION on Windows x64."
grep -E '^(numpy|scipy|pandas|matplotlib|xlwings|pywin32)==' excel/requirements-windows.txt | sed 's/ \\$//'
