@echo off
rem Extraplus for Excel: install or update for the current user (no admin rights needed).
rem Double-click this file. The work is done by files\install.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0files\install.ps1" %*
if %errorlevel% neq 0 if %errorlevel% neq 10 (
    echo.
    echo PowerShell could not run files\install.ps1 ^(exit code %errorlevel%^).
    pause
)
