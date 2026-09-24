#!/usr/bin/env bash
# Build dist/extraplus-excel-vX.Y.Z.zip, the file users download.
#
# The zip is made from the committed HEAD with `git archive` (so .gitattributes
# export-ignore applies: no research/, tests/, legacy/ or excel/build/), with
# the extraplus wheel built from that same source. Layout:
#
#   extraplus-excel-vX.Y.Z/
#       install.bat  uninstall.bat  README.md
#       files/  install.ps1 uninstall.ps1 uninstall.bat extraplus.xlam
#               extrapolation.py requirements-windows.txt pins.psd1
#               extraplus-X.Y.Z-py3-none-any.whl manifest.json
#
# Usage: excel/build/make_release.sh [--dry-run]
#   --dry-run  allow a missing/stale extraplus.xlam (a placeholder is packed)
#              and write dist/extraplus-excel-vX.Y.Z-DRYRUN.zip.
# Needs uv in WSL (UV=/path/to/uv if it is not on PATH).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
UV="${UV:-uv}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

if [ "$DRY_RUN" = 0 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "The working tree has uncommitted changes. Commit them first: the release is built from HEAD." >&2
    exit 1
fi

VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
NAME="extraplus-excel-v$VERSION"
[ "$DRY_RUN" = 1 ] && ZIP="dist/$NAME-DRYRUN.zip" || ZIP="dist/$NAME.zip"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"
git archive --format=tar HEAD | tar -x -C "$WORK/src"
SRC="$WORK/src"
echo "==> $NAME from $(git rev-parse --short HEAD)"

# The add-in must match the UDF module and the pinned xlwings (its VBA loads
# xlwings64-<version>.dll by name).
if ! python3 - "$SRC" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
src = Path(sys.argv[1]) / "excel"
problems = []
if not (src / "extraplus.xlam").exists():
    problems.append("excel/extraplus.xlam is not committed")
info_file = src / "extraplus.xlam.buildinfo.json"
if info_file.exists():
    info = json.loads(info_file.read_text())
    sha = hashlib.sha256((src / "extrapolation.py").read_bytes()).hexdigest()
    lock = re.search(r"^xlwings==(\S+)", (src / "requirements-windows.txt").read_text(), re.M).group(1)
    if info["extrapolation_py_sha256"] != sha:
        problems.append("excel/extrapolation.py changed after extraplus.xlam was built")
    if info["xlwings"] != lock:
        problems.append(f"extraplus.xlam was built with xlwings {info['xlwings']}, the lock pins {lock}")
else:
    problems.append("excel/extraplus.xlam.buildinfo.json is missing")
if problems:
    print("\n".join("    " + p for p in problems), file=sys.stderr)
    sys.exit(1)
PY
then
    if [ "$DRY_RUN" = 1 ]; then
        echo "    (dry run: packing a placeholder extraplus.xlam)"
        [ -f "$SRC/excel/extraplus.xlam" ] || : > "$SRC/excel/extraplus.xlam"
    else
        echo "Rebuild the add-in with excel/build/build_addin.sh and commit it." >&2
        exit 1
    fi
fi

"$UV" build --quiet --wheel --out-dir "$WORK/wheel" "$SRC"
WHEEL="$WORK/wheel/extraplus-$VERSION-py3-none-any.whl"
[ -f "$WHEEL" ] || { echo "Expected wheel $WHEEL was not built." >&2; exit 1; }

STAGE="$WORK/stage/$NAME"
mkdir -p "$STAGE/files"
cp "$SRC"/excel/{install.bat,uninstall.bat} "$STAGE/"
cp "$SRC/excel/README.md" "$STAGE/README.md"
cp "$SRC"/excel/{install.ps1,uninstall.ps1,uninstall.bat,extraplus.xlam,extrapolation.py,requirements-windows.txt,pins.psd1} \
   "$WHEEL" "$STAGE/files/"
python3 "$REPO/excel/build/write_manifest.py" "$STAGE/files/$(basename "$WHEEL")" \
    "$STAGE/files/requirements-windows.txt" "$STAGE/files/manifest.json" >/dev/null

mkdir -p dist
python3 - "$WORK/stage" "$NAME" "$ZIP" <<'PY'
import sys, zipfile
from pathlib import Path
stage, name, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = sorted(p for p in (stage / name).rglob("*") if p.is_file())
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(stage).as_posix())

# Checks on what users get.
names = zipfile.ZipFile(out).namelist()
inner = [n.split("/", 1)[1] for n in names]
required = ["install.bat", "uninstall.bat", "README.md"] + [f"files/{f}" for f in (
    "install.ps1", "uninstall.ps1", "uninstall.bat", "extraplus.xlam", "extrapolation.py",
    "requirements-windows.txt", "pins.psd1", "manifest.json")]
missing = [r for r in required if r not in inner]
forbidden = [n for n in inner if n.split("/")[0] in {"research", "tests", "legacy", "dist", "build", "src"}]
not_crlf = []
with zipfile.ZipFile(out) as z:
    for n in names:
        if n.endswith((".bat", ".ps1", ".psd1")):
            data = z.read(n)
            if data.count(b"\n") != data.count(b"\r\n"):
                not_crlf.append(n)
for label, bad in (("missing", missing), ("must not be shipped", forbidden), ("not CRLF", not_crlf)):
    if bad:
        sys.exit(f"Release check failed, {label}: {bad}")
print(f"==> Wrote {out} ({out.stat().st_size // 1024} KiB, {len(names)} files)")
for n in names:
    print("    " + n)
PY
