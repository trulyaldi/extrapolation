"""Write manifest.json for install.ps1: package version, wheel file + SHA256,
and the xlOil version pinned in requirements-windows.txt (install.ps1 checks
that exactly that xlOil, which provides extraplus.xll, was installed)."""

import hashlib
import json
import re
import sys
from pathlib import Path


def locked_xloil(lock):
    match = re.search(r"^xloil==(\S+)", Path(lock).read_text(), re.MULTILINE | re.IGNORECASE)
    if not match:
        raise SystemExit(f"xlOil is not pinned in {lock}")
    return match.group(1)


def main(wheel, lock, out):
    wheel = Path(wheel)
    name, version = wheel.name.split("-")[:2]
    manifest = {
        "version": version,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "xloil": locked_xloil(lock),
    }
    Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
