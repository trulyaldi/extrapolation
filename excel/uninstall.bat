@echo off
rem Extraplus for Excel: remove it for the current user.
rem Runs a temporary copy of install.ps1 so this folder can be deleted, and does
rem everything in one parenthesized block because cmd.exe re-reads a batch file
rem from disk after each command and this file is deleted along the way.
setlocal
set "EXTRAPLUS_SETUP=%~dp0install.ps1"
if not exist "%EXTRAPLUS_SETUP%" set "EXTRAPLUS_SETUP=%~dp0files\install.ps1"
set "EXTRAPLUS_TEMP=%TEMP%\extraplus-uninstall-%RANDOM%.ps1"
(
    copy /y "%EXTRAPLUS_SETUP%" "%EXTRAPLUS_TEMP%" >nul
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%EXTRAPLUS_TEMP%" -Uninstall %*
    del "%EXTRAPLUS_TEMP%" >nul 2>&1
    exit /b
)
