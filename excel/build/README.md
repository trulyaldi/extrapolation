# Building the Excel release (maintainers)

Users only get the release zip. Everything in this folder runs on the maintainer's
machine and is left out of the zip (`export-ignore`).

## Pieces

| File | Runs on | Purpose |
|---|---|---|
| `lock.sh` | WSL | regenerate `excel/requirements-windows.txt` (hashed Windows lock) |
| `constraints.txt` | | core library versions for the lock (aligned with the research env) |
| `build_addin.sh` | WSL, drives Windows | rebuild `excel/extraplus.xlam` + `.buildinfo.json` |
| `build_addin.ps1`, `build_addin.py` | Windows | build environment and the Excel automation |
| `write_manifest.py` | WSL | `manifest.json` (wheel checksum, pinned xlwings) for `install.ps1` |
| `make_release.sh` | WSL | `dist/extraplus-excel-vX.Y.Z.zip` from the committed HEAD |
| `e2e_excel.py`, `e2e_compare.py` | Windows / WSL | end-to-end test in a real Excel |

Pins that must move together: `excel/pins.psd1` (uv + Python), the lock, and the
add-in (its VBA loads `xlwings64-<version>.dll`, so it must be rebuilt whenever the
locked xlwings changes; `make_release.sh` refuses otherwise).

## Prerequisites

- uv in WSL (`UV=/path/to/uv` if it is not on PATH).
- Build machine only: desktop Excel with **File > Options > Trust Center > Trust
  Center Settings > Macro Settings > Trust access to the VBA project object model**.
  Users never need this.

## Release steps

```bash
excel/build/lock.sh               # only when changing dependency versions
excel/build/build_addin.sh        # whenever extrapolation.py, the lock or pins change
git add excel/extraplus.xlam excel/extraplus.xlam.buildinfo.json && git commit
excel/build/make_release.sh       # -> dist/extraplus-excel-vX.Y.Z.zip
```

The version comes from `pyproject.toml`. `build_addin.sh` works in
`%USERPROFILE%\extraplus-build` (Windows programs must not run from `\\wsl$`), with
its own uv/Python/venv under `env\`; it does not touch the user install, Excel's
registry settings, or `%LOCALAPPDATA%`.

## How the add-in finds Python

`extraplus.xlam` is xlwings' custom add-in template (`quickstart_addin.xlam`, which
embeds the xlwings VBA) renamed to the `extraplus` project. Its hidden
`extraplus.conf` sheet pins every setting xlwings reads, with paths relative to
`%EXTRAPLUS_HOME%`, which `install.ps1` sets per user. A `Workbook_Open` handler
copies `EXTRAPLUS_HOME` from `HKCU\Environment` into Excel's environment when Excel
was started by a program that predates the install.
