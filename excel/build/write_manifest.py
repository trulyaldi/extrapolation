"""Write manifest.json for install.ps1: package version, wheel file + SHA256,
and the xlwings version pinned in requirements-windows.txt (the add-in's VBA
only works with exactly that xlwings)."""

import hashlib
import json
import re
import sys
from pathlib import Path


def locked_xlwings(lock):
    match = re.search(r"^xlwings==(\S+)", Path(lock).read_text(), re.MULTILINE)
    if not match:
        raise SystemExit(f"xlwings is not pinned in {lock}")
    return match.group(1)


def main(wheel, lock, out):
    wheel = Path(wheel)
    name, version = wheel.name.split("-")[:2]
    manifest = {
        "version": version,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "xlwings": locked_xlwings(lock),
    }
    Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
