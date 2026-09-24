# Extraplus for Excel

THIS MARKDOWN FILE CONTAINS ALL THE NECESSARY INFORMATION. please, read it carefully. 

Extraplus adds extrapolation functions to Excel on Windows:

```
=EXTRAP1(A2:A20, B2:B20)
```

The functions work in **macro-enabled workbooks (`.xlsm`)**. The workbook can
be saved in any folder, and you can create new ones anywhere. You don't need
Python, admin rights, or any Excel settings.

## Install

1. Close Excel.
2. Unzip `extraplus-excel-vX.Y.Z.zip` (right-click > Extract All).
3. Double-click `install.bat`. If Windows says "Windows protected your PC",
   click **More info > Run anyway**.
4. Wait for `SUCCESS` (a few minutes the first time), then press Enter.

You can delete the unzipped folder afterwards. The first calculation after
starting Excel takes a few seconds.

## Functions

`x` is the range of basis sizes and `y` the range of values. The digit selects
the model: **1** exponential, **2** square-root exponential, **3** power law.

| Function | Returns |
|---|---|
| `EXTRAP1(x, y, [b_fixed], [n_fit])` | extrapolated value C at infinite basis |
| `UNCERTAINTY1(x, y, [b_fixed], [n_fit])` | uncertainty of C |
| `DECAY_RATE1(x, y, [b_fixed], [n_fit])` | decay parameter B |
| `EXTRAP1_V(...)` / `EXTRAP1_H(...)` | C, uncertainty and B in 3 cells, down / across |
| `EXTRAP1_PLOT(...)` | plot of the data, fit and extrapolated value |
| `EXTRAP1_PLOT_LOG(...)` | plot of the fit in linearized form |
| `EXTRAPLUS_VERSION()` | installed version |

Every function also exists for models 2 and 3 (`EXTRAP2`, `EXTRAP3`, ...).

Optional arguments:
- `b_fixed`: use this value for B instead of fitting it.
- `n_fit`: fit only the first `n_fit` points. Useful for checking whether the
  extrapolation is stable as more points are added.

If the input is invalid, the cell shows a message such as
`Error: At least three paired numeric x/y observations are required.`

## Update

Close Excel and run `install.bat` from the new zip. Your workbooks are not
affected.

## Uninstall

Close Excel, then go to **Settings > Apps > Extraplus for Excel > Uninstall**.

## Troubleshooting

- **"Microsoft Excel is running"**: close all Excel windows and try again.
- **`#NAME?` in a cell**: the add-in isn't loaded. Check that **Extraplus** is
  ticked under **File > Options > Add-ins > Manage: Excel Add-ins > Go**. If
  it's missing, run `install.bat` again.
- **Anything else**: run `install.bat` again, which repairs the installation.
  If that doesn't help, send me the log file `%TEMP%\extraplus-setup.log`.

## Known issue

Opening a workbook made on someone else's computer may show a link prompt or
`#NAME?`. This hasn't been tested yet.