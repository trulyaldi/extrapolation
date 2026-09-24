# Removes Extraplus for Excel for the current user. The logic is in install.ps1
# (-Uninstall); uninstall.bat is the double-click launcher.
param(
    [string]$InstallRoot,
    [switch]$NoPause
)
& (Join-Path $PSScriptRoot 'install.ps1') -Uninstall @PSBoundParameters
exit $LASTEXITCODE
