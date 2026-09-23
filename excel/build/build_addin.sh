#!/usr/bin/env bash
# Build (or rebuild) excel/extraplus.xlam. Run from WSL on the build machine.
#
# Needs: desktop Excel on the Windows side with "Trust access to the VBA
# project object model" enabled, uv in WSL (UV=/path/to/uv if not on PATH),
# and internet access. Excel should be closed; a hidden Excel instance is used.
# Works in %USERPROFILE%\extraplus-build (override: EXTRAPLUS_BUILD_DIR, a
# Windows path) because Windows programs must not run from \\wsl$ paths.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
UV="${UV:-uv}"

WIN_PROFILE="$(cd /mnt/c && cmd.exe /d /c 'echo %USERPROFILE%' | tr -d '\r')"
BUILD_WIN="${EXTRAPLUS_BUILD_DIR:-$WIN_PROFILE\\extraplus-build}"
BUILD="$(wslpath -u "$BUILD_WIN")"
STAGE="$BUILD/stage"

rm -rf "$STAGE"
mkdir -p "$STAGE"
echo "==> Staging in $BUILD_WIN\\stage"
"$UV" build --quiet --wheel --out-dir "$STAGE" "$REPO"
python3 "$REPO/excel/build/write_manifest.py" "$(ls "$STAGE"/extraplus-*.whl)" \
    "$REPO/excel/requirements-windows.txt" "$STAGE/manifest.json"
cp "$REPO"/excel/{install.ps1,pins.psd1,requirements-windows.txt,extrapolation.py} \
   "$REPO"/excel/build/{build_addin.ps1,build_addin.py} "$STAGE/"

echo "==> Building on Windows"
(cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$BUILD_WIN\\stage\\build_addin.ps1")

cp "$STAGE/out/extraplus.xlam" "$STAGE/out/extraplus.xlam.buildinfo.json" "$REPO/excel/"
echo "==> Updated excel/extraplus.xlam and excel/extraplus.xlam.buildinfo.json"
