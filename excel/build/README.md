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

## End-to-end test notes (v0.1.0)

Excel 2021 (16.0.14332.20447, 64-bit) on Windows 10 21H2 (19044), with a global
xlwings add-in in XLSTART; release zip extracted with Mark-of-the-Web on every file.

| Check | Result |
|---|---|
| install.bat (fresh, and again as update) | SUCCESS; one `OPEN` value and one trusted location after each run and after each Excel quit |
| EXTRAP1-3, UNCERTAINTY1-3, DECAY_RATE1-3 in a new .xlsx | bit-identical to extraplus in the installed venv and in the research .venv |
| EXTRAPLUS_VERSION(), bad input | `0.1.0`; `Error: At least three paired numeric x/y observations are required.` |
| Reopen the saved .xlsx (before and after the update) | recalculates, same values, no dialog |
| UDF server | `%EXTRAPLUS_HOME%\venv\Scripts\pythonw.exe` (+ base `python\...\pythonw.exe`) only |
| EXTRAPLUS_HOME missing from Excel's environment | picked up by the add-in's registry fallback |
| EXTRAP1_V / EXTRAP1_H | legacy array output; dynamic-array spill not tested via COM (every `Range.Formula2` write and `HasSpill` read fails with 0x800A03EC on this Excel) |
| Uninstall (the registered UninstallString) | exit code 0; `OPEN`, trusted location, EXTRAPLUS_HOME, Apps & features entry and folder removed; Excel then starts with no dialog |
| Manual check: freshly started Excel inherits EXTRAPLUS_HOME | not required (covered by the registry fallback) |
| Manual check: dynamic-array spill in the Excel UI | not required (spill is out of scope for v0.1.0) |

Not tested: 32-bit Office, other Excel versions, the `_PLOT` functions, a second
Windows account (saved workbooks link to the add-in's absolute path,
`xl/externalLinks/`).
