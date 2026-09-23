@echo off
setlocal EnableExtensions

REM ============================================================
REM Extraplus Excel Setup
REM Supported Python versions: 3.11 - 3.14
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo                   EXTRAPLUS SETUP
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Check required project files
REM ------------------------------------------------------------

echo [1/6] Checking project files...
echo.

if not exist "extrap.py" (
    echo ERROR: extrap.py was not found.
    echo Make sure setup.bat is in the root Extraplus folder.
    goto :error
)

if not exist "extrapolation.py" (
    echo ERROR: extrapolation.py was not found.
    echo Make sure setup.bat is in the root Extraplus folder.
    goto :error
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found.
    echo Make sure requirements.txt is in the root Extraplus folder.
    goto :error
)

echo Project files found.
echo.

REM ------------------------------------------------------------
REM 2. Find a supported Python installation
REM ------------------------------------------------------------

echo [2/6] Checking Python...
echo.

set "PYTHON_CMD="

REM Prefer newest explicitly supported version.

py -3.14 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.14"
    goto :python_found
)

py -3.13 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.13"
    goto :python_found
)

py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    goto :python_found
)

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto :python_found
)

REM Fall back to python if launcher is unavailable.

python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1

if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)

echo ERROR: A supported Python version was not found.
echo.
echo Extraplus currently supports:
echo.
echo     Python 3.11
echo     Python 3.12
echo     Python 3.13
echo     Python 3.14
echo.
echo Please install one of these Python versions and run setup.bat again.
echo.
echo During Python installation, enabling:
echo.
echo     Add Python to PATH
echo.
echo is recommended.
echo.
goto :error


:python_found

echo Compatible Python installation found:
%PYTHON_CMD% --version
echo.

REM ------------------------------------------------------------
REM 3. Create or validate virtual environment
REM ------------------------------------------------------------

echo [3/6] Setting up Python virtual environment...
echo.

if exist ".venv\Scripts\python.exe" (

    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1

    if errorlevel 1 (
        echo Existing .venv uses an unsupported Python version.
        echo Recreating .venv...
        echo.

        rmdir /s /q ".venv"

        if exist ".venv" (
            echo ERROR: Could not remove the existing .venv folder.
            echo.
            echo Close Excel, Command Prompt, VS Code, Python,
            echo or any program using the environment.
            echo Then run setup.bat again.
            goto :error
        )

        %PYTHON_CMD% -m venv .venv

        if errorlevel 1 (
            echo ERROR: Could not recreate the Python virtual environment.
            goto :error
        )

    ) else (
        echo Existing .venv found. Reusing it.
    )

) else (

    echo Creating .venv...
    echo.

    %PYTHON_CMD% -m venv .venv

    if errorlevel 1 (
        echo ERROR: Could not create the Python virtual environment.
        goto :error
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The virtual environment was not created correctly.
    goto :error
)

echo.
echo Virtual environment Python:
".venv\Scripts\python.exe" --version
echo.

REM ------------------------------------------------------------
REM 4. Install Python dependencies
REM ------------------------------------------------------------

echo [4/6] Installing Python dependencies...
echo.

".venv\Scripts\python.exe" -m pip install --upgrade pip

if errorlevel 1 (
    echo.
    echo ERROR: Could not upgrade pip.
    goto :error
)

echo.

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Could not install Python dependencies.
    echo.
    echo Check:
    echo     - Your internet connection
    echo     - requirements.txt
    echo     - Your Python installation
    goto :error
)

echo.
echo Python dependencies installed successfully.
echo.

REM ------------------------------------------------------------
REM 5. Verify Python packages and Extraplus modules
REM ------------------------------------------------------------

echo [5/6] Verifying Extraplus installation...
echo.

".venv\Scripts\python.exe" -c "import numpy, pandas, scipy, matplotlib, xlwings; import extrap; import extrapolation; print('Python imports OK'); print('xlwings version:', xlwings.__version__)"

if errorlevel 1 (
    echo.
    echo ERROR: Extraplus Python verification failed.
    echo.
    echo One or more required packages or project modules could not be imported.
    echo.
    echo Check the Python traceback shown above.
    goto :error
)

echo.
echo Python verification successful.
echo.

REM ------------------------------------------------------------
REM 6. Install matching xlwings Excel add-in
REM ------------------------------------------------------------

echo [6/6] Installing xlwings Excel add-in...
echo.

if not exist ".venv\Scripts\xlwings.exe" (
    echo ERROR: xlwings.exe was not found in the virtual environment.
    echo.
    echo Make sure xlwings is included in requirements.txt.
    goto :error
)

REM Do not modify Excel add-ins while Excel is running.

tasklist /FI "IMAGENAME eq EXCEL.EXE" 2>nul | find /I "EXCEL.EXE" >nul

if not errorlevel 1 (
    echo.
    echo ERROR: Microsoft Excel is currently running.
    echo.
    echo Close ALL Excel windows before installing the xlwings add-in.
    echo Then run setup.bat again.
    goto :error
)

echo Removing any previously installed xlwings add-in...
echo.

".venv\Scripts\xlwings.exe" addin remove >nul 2>&1

echo Installing xlwings add-in from this project's virtual environment...
echo.

".venv\Scripts\xlwings.exe" addin install

if errorlevel 1 (
    echo.
    echo ERROR: xlwings add-in installation failed.
    echo.
    echo Make sure:
    echo     - All Excel windows are closed
    echo     - Microsoft Excel is installed
    echo     - xlwings is installed in .venv
    echo.
    echo Then run setup.bat again.
    goto :error
)

echo.
echo xlwings Excel add-in installed successfully.
echo.

REM ------------------------------------------------------------
REM Final verification
REM ------------------------------------------------------------

echo Verifying xlwings version used by this project...
echo.

".venv\Scripts\python.exe" -c "import xlwings; print('xlwings Python package version:', xlwings.__version__)"

if errorlevel 1 (
    echo.
    echo ERROR: Could not verify xlwings after installation.
    goto :error
)

REM ------------------------------------------------------------
REM Finished
REM ------------------------------------------------------------

echo.
echo ============================================================
echo              EXTRAPLUS INSTALLATION COMPLETE
echo ============================================================
echo.
echo Project folder:
echo.
echo     %CD%
echo.
echo Virtual environment:
echo.
echo     %CD%\.venv
echo.
echo Python interpreter:
echo.
echo     %CD%\.venv\Scripts\python.exe
echo.
echo Python GUI interpreter:
echo.
echo     %CD%\.venv\Scripts\pythonw.exe
echo.
echo ============================================================
echo                 EXCEL CONFIGURATION
echo ============================================================
echo.
echo Complete the following steps in Excel:
echo.
echo 1. Open your .xlsm workbook.
echo.
echo 2. Open the xlwings ribbon.
echo.
echo 3. Set Interpreter to:
echo.
echo     %CD%\.venv\Scripts\pythonw.exe
echo.
echo 4. Set UDF Modules to:
echo.
echo     extrapolation
echo.
echo 5. Enable:
echo.
echo     Add Workbook to PYTHONPATH
echo.
echo 6. Go to:
echo.
echo     File
echo     ^> Options
echo     ^> Trust Center
echo     ^> Trust Center Settings
echo     ^> Macro Settings
echo.
echo    Enable:
echo.
echo     Trust access to the VBA project object model
echo.
echo 7. Press:
echo.
echo     Alt + F11
echo.
echo    Then go to:
echo.
echo     Tools ^> References
echo.
echo    Make sure:
echo.
echo     xlwings
echo.
echo    is enabled.
echo.
echo 8. Return to Excel.
echo.
echo 9. Open the xlwings ribbon.
echo.
echo 10. Click:
echo.
echo     Import Python UDFs
echo.
echo 11. Click:
echo.
echo     Restart UDF Server
echo.
echo 12. Save the workbook as an .xlsm file.
echo.
echo ============================================================
echo                     TEST FUNCTIONS
echo ============================================================
echo.
echo Numerical functions:
echo.
echo     =EXTRAP1(A2:A20,B2:B20)
echo     =UNCERTAINTY1(A2:A20,B2:B20)
echo     =DECAY_RATE1(A2:A20,B2:B20)
echo.
echo Model families:
echo.
echo     EXTRAP1 / EXTRAP2 / EXTRAP3
echo     UNCERTAINTY1 / UNCERTAINTY2 / UNCERTAINTY3
echo     DECAY_RATE1 / DECAY_RATE2 / DECAY_RATE3
echo.
echo Plot functions:
echo.
echo     EXTRAP1_PLOT
echo     EXTRAP2_PLOT
echo     EXTRAP3_PLOT
echo.
echo     EXTRAP1_PLOT_LOG
echo     EXTRAP2_PLOT_LOG
echo     EXTRAP3_PLOT_LOG
echo.
echo ============================================================
echo.
echo Setup completed successfully.
echo.
pause
exit /b 0


:error

echo.
echo ============================================================
echo                    SETUP FAILED
echo ============================================================
echo.
echo Please fix the error shown above and run setup.bat again.
echo.
pause
exit /b 1