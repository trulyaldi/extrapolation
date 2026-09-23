# Extraplus for Excel

This repository installs the Extraplus Python UDFs into Microsoft Excel on Windows.

The UDFs are intended to be imported into **Excel Macro-Enabled Workbooks (`.xlsm`)**.

After installation, functions such as the following are available in Excel:

```excel
=EXTRAP1(A2:A20,B2:B20)
=UNCERTAINTY1(A2:A20,B2:B20)
=DECAY_RATE1(A2:A20,B2:B20)
```

## Requirements

* Windows
* Microsoft Excel Desktop
* Python 3.11 or newer
* An Excel Macro-Enabled Workbook (`.xlsm`)
* Internet connection during the initial installation

## Repository Files

Keep these files together in the same folder:

```text
extraplus-excel/
├── extrap.py
├── extrapolation.py
├── requirements.txt
├── setup.bat
├── README.md
└── your-workbook.xlsm
```

Your `.xlsm` workbook can have any filename.

For example:

```text
calculation.xlsm
results.xlsm
energy_extrapolation.xlsm
```

---

# Installation

## 1. Install Python 3.11 or Newer

Install **Python 3.11 or newer**.

During installation, enable:

```text
Add Python to PATH
```

To verify the installation, open Command Prompt and run:

```bat
py -3 --version
```

You should see Python 3.11 or newer, for example:

```text
Python 3.11.x
```

or:

```text
Python 3.12.x
```

---

## 2. Download Extraplus

Either clone this repository:

```bat
git clone <repository-url>
```

or download it from GitHub:

```text
Code
→ Download ZIP
```

If you download the ZIP, extract it before continuing.

Do not run Extraplus directly from inside the ZIP file.

---

## 3. Prepare Your Excel Workbook

The workbook must be an **Excel Macro-Enabled Workbook (`.xlsm`)**.

If your workbook is currently `.xlsx`, open it in Excel and save it as:

```text
File
→ Save As
→ Excel Macro-Enabled Workbook (*.xlsm)
```

For example:

```text
calculation.xlsx
```

should become:

```text
calculation.xlsm
```

Place the `.xlsm` workbook in the same folder as:

```text
extrap.py
extrapolation.py
setup.bat
requirements.txt
```

For example:

```text
extraplus-excel/
├── extrap.py
├── extrapolation.py
├── requirements.txt
├── setup.bat
└── calculation.xlsm
```

---

## 4. Close Excel

Close **all Excel windows** before continuing.

---

## 5. Run `setup.bat`

Double-click:

```text
setup.bat
```

The script will automatically:

1. Check for Python 3.11 or newer.
2. Create a local `.venv` Python environment.
3. Install the required Python packages.
4. Install the xlwings Excel add-in.

Wait until the script reports that the installation is complete.

---

# Configure the `.xlsm` Workbook

## 6. Open Your `.xlsm` File

Open your macro-enabled workbook.

For example:

```text
calculation.xlsm
```

You should see an **xlwings** tab in the Excel ribbon.

If the xlwings tab does not appear:

1. Close Excel.
2. Open Command Prompt in the Extraplus folder.
3. Run:

```bat
.venv\Scripts\xlwings.exe addin install
```

4. Reopen the `.xlsm` workbook.

---

## 7. Set the Python Interpreter

Open the **xlwings** tab.

Set **Interpreter** to:

```text
<Extraplus folder>\.venv\Scripts\python.exe
```

For example:

```text
C:\Users\YourName\Documents\extraplus-excel\.venv\Scripts\python.exe
```

---

## 8. Set the UDF Module

In the xlwings ribbon, set:

```text
UDF Modules
```

to:

```text
extrapolation
```

Do not enter:

```text
extrapolation.py
```

Use only:

```text
extrapolation
```

---

## 9. Enable Workbook PYTHONPATH

In the xlwings ribbon, enable:

```text
Add Workbook to PYTHONPATH
```

This allows Python to find:

```text
extrapolation.py
extrap.py
```

in the same folder as the `.xlsm` workbook.

---

## 10. Enable VBA Project Access

In Excel go to:

```text
File
→ Options
→ Trust Center
→ Trust Center Settings
→ Macro Settings
```

Enable:

```text
Trust access to the VBA project object model
```

Click **OK**.

---

## 11. Enable the xlwings VBA Reference

Press:

```text
Alt + F11
```

to open the VBA editor.

Then go to:

```text
Tools
→ References
```

Find:

```text
xlwings
```

and tick the checkbox:

```text
☑ xlwings
```

Click **OK**.

This step is required.

If the xlwings reference is not enabled, Excel may display:

```text
Object required
```

when using the UDFs.

---

## 12. Import the Python UDFs

Return to the `.xlsm` workbook.

Open the **xlwings** tab and click:

```text
Import Python UDFs
```

Wait for the import to finish.

Then click:

```text
Restart UDF Server
```

Save the `.xlsm` workbook afterward so that the imported VBA wrappers are preserved.

---

# Test the Installation

Suppose:

```text
A2:A20 = x values
B2:B20 = y values
```

Test:

```excel
=EXTRAP1(A2:A20,B2:B20)
```

You can also test:

```excel
=UNCERTAINTY1(A2:A20,B2:B20)
```

and:

```excel
=DECAY_RATE1(A2:A20,B2:B20)
```

The available model families are:

```excel
=EXTRAP1(...)
=EXTRAP2(...)
=EXTRAP3(...)

=UNCERTAINTY1(...)
=UNCERTAINTY2(...)
=UNCERTAINTY3(...)

=DECAY_RATE1(...)
=DECAY_RATE2(...)
=DECAY_RATE3(...)
```

---

# Using Extraplus in Another `.xlsm` Workbook

For every additional `.xlsm` workbook that should use Extraplus:

1. Place the `.xlsm` file in the same folder as:

```text
extrap.py
extrapolation.py
```

2. Open the `.xlsm` workbook.

3. In the xlwings ribbon, set:

```text
Interpreter = <Extraplus folder>\.venv\Scripts\python.exe
```

4. Set:

```text
UDF Modules = extrapolation
```

5. Enable:

```text
Add Workbook to PYTHONPATH
```

6. Press:

```text
Alt + F11
```

7. Go to:

```text
Tools
→ References
```

8. Tick:

```text
☑ xlwings
```

9. Return to Excel.

10. Click:

```text
Import Python UDFs
```

11. Click:

```text
Restart UDF Server
```

12. Save the `.xlsm` workbook.

---

# Troubleshooting

## `Object required`

Press:

```text
Alt + F11
```

Then go to:

```text
Tools
→ References
```

Make sure:

```text
☑ xlwings
```

is checked.

After enabling it, return to Excel and click:

```text
Import Python UDFs
```

followed by:

```text
Restart UDF Server
```

---

## xlwings Is Missing from References

Close Excel completely.

Open Command Prompt in the Extraplus folder and run:

```bat
.venv\Scripts\xlwings.exe addin install
```

Reopen the `.xlsm` workbook.

Then:

```text
Alt + F11
→ Tools
→ References
→ ☑ xlwings
```

---

## `#NAME?`

If Excel displays:

```text
#NAME?
```

check that:

```text
UDF Modules = extrapolation
```

Then click:

```text
Import Python UDFs
```

again.

Make sure the workbook is saved as:

```text
.xlsm
```

---

## `ModuleNotFoundError`

Make sure the `.xlsm` workbook is in the same folder as:

```text
extrap.py
extrapolation.py
```

For example:

```text
extraplus-excel/
├── extrap.py
├── extrapolation.py
└── calculation.xlsm
```

Also make sure:

```text
Add Workbook to PYTHONPATH
```

is enabled.

---

## xlwings Tab Is Missing

Close Excel and run:

```bat
.venv\Scripts\xlwings.exe addin install
```

Then reopen Excel.

---

## Macros Are Blocked

If Windows blocks a downloaded `.xlsm` workbook:

1. Close Excel.
2. Right-click the `.xlsm` file.
3. Select **Properties**.
4. If **Unblock** appears, tick it.
5. Click **Apply**.
6. Reopen the workbook.

---

# Updating Extraplus

After downloading or pulling a newer version:

```bat
git pull
```

If only the Python fitting code changed:

```text
xlwings
→ Restart UDF Server
```

If the Excel UDF names or arguments changed:

```text
xlwings
→ Import Python UDFs
```

again.

If `requirements.txt` changed:

```text
setup.bat
```

should be run again.
