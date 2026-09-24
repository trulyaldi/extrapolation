@echo off
rem Extraplus for Excel: remove it for the current user.
rem Runs a temporary copy of install.ps1 so this folder can be deleted. The last
rem line starts with (goto), which ends the batch file before this folder (and
rem this file) is deleted, so cmd.exe does not try to read it again; the rest of
rem that line has already been read and still runs, and passes on the exit code
rem of install.ps1.
rem Leave the install folder first, so it is not the working directory. This comes
rem before setlocal because the batch file ending restores the directory setlocal saved.
cd /d "%TEMP%"
setlocal
set "EXTRAPLUS_SETUP=%~dp0install.ps1"
if not exist "%EXTRAPLUS_SETUP%" set "EXTRAPLUS_SETUP=%~dp0files\install.ps1"
set "EXTRAPLUS_TEMP=%TEMP%\extraplus-uninstall-%RANDOM%.ps1"
copy /y "%EXTRAPLUS_SETUP%" "%EXTRAPLUS_TEMP%" >nul
(goto) 2>nul & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%EXTRAPLUS_TEMP%" -Uninstall %* & call set "EXTRAPLUS_RC=%%errorlevel%%" & del "%EXTRAPLUS_TEMP%" >nul 2>&1 & call cmd /d /c exit %%EXTRAPLUS_RC%%
