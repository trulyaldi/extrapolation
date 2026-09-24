<#
    Windows side of build_addin.sh: sets up a build environment with the same
    pins and install code as the user installer, then runs build_addin.py.
    Everything goes into -EnvRoot (default: ..\env next to the stage folder);
    nothing is registered in Excel or the registry.
#>
param(
    [string]$EnvRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'env')
)

. (Join-Path $PSScriptRoot 'install.ps1')   # functions only (the entry point is skipped)

$pins = Import-PowerShellDataFile -LiteralPath (Join-Path $PSScriptRoot 'pins.psd1')
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json

try {
    New-Item -ItemType Directory -Force -Path $EnvRoot | Out-Null
    Set-UvEnvironment $EnvRoot
    Write-Step "Build environment: $EnvRoot"
    $uv = Install-Uv $EnvRoot $pins
    $venv = Install-PythonEnvironment $EnvRoot $uv $pins $manifest

    Write-Step 'Building extraplus.xlam'
    $python = Join-Path $venv 'Scripts\python.exe'
    Invoke-Native $python @((Join-Path $PSScriptRoot 'build_addin.py'), '--out', (Join-Path $PSScriptRoot 'out'), '--udf-dir', $PSScriptRoot)
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
