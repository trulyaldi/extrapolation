<#
.SYNOPSIS
    Installs, updates or removes Extraplus for Excel for the current user.

.DESCRIPTION
    Run through install.bat (install or update) or uninstall.bat. No admin
    rights are needed; everything goes into one per-user folder plus a few
    HKCU registry values:

      <install folder>\bin\uv.exe        pinned uv (checksum-verified)
      <install folder>\python\           uv-managed Python
      <install folder>\venv\             Python environment with extraplus
      <install folder>\udf\              the Excel UDF module (extrapolation.py)
      <install folder>\extraplus.xlam    the Excel add-in
      <install folder>\cache\            uv download cache

    HKCU\Environment\EXTRAPLUS_HOME       install folder, read by the add-in
    Excel\Options\OPENn                   loads the add-in when Excel starts
    Excel\Security\Trusted Locations      trusts the install folder
    ...\CurrentVersion\Uninstall\Extraplus  "Apps & features" entry

    The install folder is %LOCALAPPDATA%\Extraplus. xlwings passes paths to
    Windows as ANSI text, so if that path contains non-ASCII characters (for
    example a Cyrillic user name) C:\Extraplus-u<id> is used instead, with
    permissions limited to the user (a folder in C:\ is otherwise writable by
    every user of the PC).

    This file is ASCII only, for Windows PowerShell 5.1.
#>
[CmdletBinding()]
param(
    # Install into this folder instead of the default (advanced; testing).
    [string]$InstallRoot,
    # Remove Extraplus instead of installing it.
    [switch]$Uninstall,
    # Do not wait for Enter at the end.
    [switch]$NoPause
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with a progress bar

$AppName = 'Extraplus'
$AppDisplayName = 'Extraplus for Excel'
$EnvVarName = 'EXTRAPLUS_HOME'
$MarkerFile = 'install-info.json'
$AddInFile = 'extraplus.xlam'
$ExcelSubKey = 'Software\Microsoft\Office\16.0\Excel'
$UninstallSubKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\Extraplus'
$SourceDir = $PSScriptRoot
$FailureExitCode = 10


# ---------------------------------------------------------------------------
# Output and process helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Write-Info([string]$Text) {
    Write-Host "    $Text"
}

function Write-Warn([string]$Text) {
    Write-Host "    WARNING: $Text" -ForegroundColor Yellow
}

function Invoke-Native {
    # Run an external program; uv reports progress on stderr, so do not turn
    # stderr into PowerShell errors. Fail on a non-zero exit code.
    param([string]$Exe, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments | Out-Host   # show output; never return it to the caller
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "$([IO.Path]::GetFileName($Exe)) $($Arguments[0]) failed (exit code $code)."
    }
}

function Get-NativeOutput {
    param([string]$Exe, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Exe @Arguments 2>$null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "$([IO.Path]::GetFileName($Exe)) $($Arguments[0]) failed (exit code $code)."
    }
    return (($output | Out-String).Trim())
}

function Assert-ExcelClosed {
    # Excel rewrites its add-in registry values when it exits, and it locks the
    # add-in and the Python files while it runs.
    if (Get-Process -Name 'EXCEL' -ErrorAction SilentlyContinue) {
        throw 'Microsoft Excel is running. Close all Excel windows (if none are visible, end EXCEL.EXE in Task Manager) and run this again.'
    }
}

function Stop-ExtraplusProcesses([string]$Root) {
    # The Excel UDF server (pythonw.exe from the venv) can outlive Excel.
    $prefix = $Root.TrimEnd('\') + '\'
    foreach ($process in @(Get-Process)) {
        $path = $null
        try { $path = $process.Path } catch { }
        if ($path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            Write-Info "Stopping $($process.ProcessName) (PID $($process.Id))"
            Stop-Process -Id $process.Id -Force
        }
    }
}


# ---------------------------------------------------------------------------
# Install folder
# ---------------------------------------------------------------------------

function Test-SafePath([string]$Path) {
    # xlwings hands the interpreter and DLL paths to Windows as ANSI strings
    # and embeds them in a cmd.exe command line, so they must be printable
    # ASCII without characters that cmd, %VAR% expansion or PYTHONPATH treat
    # specially.
    if ($Path -notmatch '^[\x20-\x7E]+$') { return $false }
    if ($Path -match '[%"''&^!;]') { return $false }
    return $true
}

function Get-FallbackRoot {
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $sha = [Security.Cryptography.SHA256]::Create()
    $bytes = $sha.ComputeHash([Text.Encoding]::ASCII.GetBytes($sid))
    $id = -join ($bytes[0..4] | ForEach-Object { $_.ToString('x2') })
    return (Join-Path ($env:SystemDrive + '\') "$AppName-u$id")
}

function Protect-Folder([string]$Path) {
    # Only this user, SYSTEM and Administrators; no inherited "Authenticated
    # Users: Modify" from the drive root.
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    Invoke-Native 'icacls.exe' @($Path, '/inheritance:r', '/grant:r', "*${sid}:(OI)(CI)F", '*S-1-5-18:(OI)(CI)F', '*S-1-5-32-544:(OI)(CI)F', '/Q') | Out-Null
}

function Test-InUserProfile([string]$Path) {
    foreach ($base in $env:LOCALAPPDATA, $env:USERPROFILE) {
        if ($base -and $Path.StartsWith($base.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-DefaultRoot {
    return (Join-Path $env:LOCALAPPDATA $AppName)
}

function Resolve-InstallRoot([string]$Requested) {
    if ($Requested) {
        $root = [IO.Path]::GetFullPath($Requested).TrimEnd('\')
        if (-not (Test-SafePath $root)) {
            throw "The install folder '$root' must be a plain ASCII path without % "" ' & ^ ! ;"
        }
        return $root
    }
    $existing = [Environment]::GetEnvironmentVariable($EnvVarName, 'User')
    if ($existing -and (Test-Path -LiteralPath (Join-Path $existing $MarkerFile))) {
        return $existing.TrimEnd('\')
    }
    $default = Get-DefaultRoot
    if (Test-SafePath $default) { return $default }
    $fallback = Get-FallbackRoot
    Write-Info "The default folder $default contains characters that xlwings cannot pass to Windows."
    Write-Info "Using $fallback instead."
    return $fallback
}

function Find-InstalledRoots {
    # Every folder that may hold an installation, for uninstalling.
    $candidates = @()
    $fromEnv = [Environment]::GetEnvironmentVariable($EnvVarName, 'User')
    if ($fromEnv) { $candidates += $fromEnv.TrimEnd('\') }
    $candidates += (Get-DefaultRoot)
    $candidates += (Get-FallbackRoot)
    $roots = @()
    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath (Join-Path $candidate $MarkerFile)) -and ($roots -notcontains $candidate)) {
            $roots += $candidate
        }
    }
    return $roots
}


# ---------------------------------------------------------------------------
# uv, Python and the virtual environment
# ---------------------------------------------------------------------------

function Set-UvEnvironment([string]$Root) {
    # Keep everything uv downloads inside the install folder and ignore any
    # uv configuration or Python variables the user may have.
    $env:UV_CACHE_DIR = Join-Path $Root 'cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
    $env:UV_NO_CONFIG = '1'
    foreach ($name in 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONHOME', 'PYTHONPATH', 'UV_PYTHON', 'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL') {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
}

function Install-Uv([string]$Root, $Pins) {
    $bin = Join-Path $Root 'bin'
    $uv = Join-Path $bin 'uv.exe'
    if (Test-Path -LiteralPath $uv) {
        $current = (Get-NativeOutput $uv @('--version')) -replace '^uv\s+(\S+).*$', '$1'
        if ($current -eq $Pins.UvVersion) {
            Write-Info "uv $current is already installed."
            return $uv
        }
        Write-Info "Replacing uv $current with $($Pins.UvVersion)."
    }
    $url = "https://github.com/astral-sh/uv/releases/download/$($Pins.UvVersion)/uv-x86_64-pc-windows-msvc.zip"
    $work = Join-Path $env:TEMP ('extraplus-uv-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    try {
        $zip = Join-Path $work 'uv.zip'
        Write-Info "Downloading uv $($Pins.UvVersion) ..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $Pins.UvSha256) {
            throw "The downloaded uv does not have the expected SHA256 checksum ($hash)."
        }
        Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $work 'x')
        New-Item -ItemType Directory -Force -Path $bin | Out-Null
        Copy-Item -LiteralPath (Join-Path $work 'x\uv.exe') -Destination $uv -Force
        Unblock-File -LiteralPath $uv
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $uv
}

function Install-PythonEnvironment([string]$Root, [string]$Uv, $Pins, $Manifest) {
    Write-Step "Python $($Pins.PythonVersion)"
    Invoke-Native $Uv @('python', 'install', $Pins.PythonVersion, '--no-bin', '--no-registry')
    $python = Get-NativeOutput $Uv @('python', 'find', '--managed-python', '--no-project', $Pins.PythonVersion)
    if (-not (Test-Path -LiteralPath $python)) { throw "uv did not report a Python $($Pins.PythonVersion) interpreter." }
    Write-Info "Using $python"

    Write-Step 'Python packages'
    $venv = Join-Path $Root 'venv'
    Invoke-Native $Uv @('venv', $venv, '--python', $python, '--clear')
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    $lock = Join-Path $SourceDir 'requirements-windows.txt'
    Invoke-Native $Uv @('pip', 'install', '--python', $venvPython, '--require-hashes', '--only-binary', ':all:', '--compile-bytecode', '-r', $lock)

    $wheel = Join-Path $SourceDir $Manifest.wheel
    $wheelHash = (Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($wheelHash -ne $Manifest.wheel_sha256) { throw "$($Manifest.wheel) does not match the checksum in manifest.json." }
    Invoke-Native $Uv @('pip', 'install', '--python', $venvPython, '--no-deps', '--compile-bytecode', $wheel)
    return $venv
}


# ---------------------------------------------------------------------------
# Registry (HKCU only)
# ---------------------------------------------------------------------------

function Open-UserKey([string]$SubKey, [switch]$Create) {
    if ($Create) { return [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($SubKey) }
    return [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($SubKey, $true)
}

function Test-OurAddIn([string]$Value) {
    # OPENn values look like:  /R "C:\path\extraplus.xlam"  or  "C:\path\x.xlam"
    $path = $Value.Trim()
    while ($path -match '^/\w+\s+(.*)$') { $path = $Matches[1].Trim() }
    $path = $path.Trim('"')
    return ($path -like "*\$AddInFile") -or ($path -ieq $AddInFile)
}

function Get-OpenEntries($Key) {
    $entries = @()
    foreach ($name in $Key.GetValueNames()) {
        if ($name -match '^OPEN(\d*)$') {
            $index = 0
            if ($Matches[1]) { $index = [int]$Matches[1] }
            $entries += [pscustomobject]@{
                Index = $index
                Name  = $name
                Value = [string]$Key.GetValue($name, '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
                Kind  = $Key.GetValueKind($name)
            }
        }
    }
    return @($entries | Sort-Object Index)
}

function Get-OpenName([int]$Index) {
    if ($Index -eq 0) { return 'OPEN' }
    return "OPEN$Index"
}

function Set-OpenEntries($Key, $Entries) {
    # Excel reads OPEN, OPEN1, OPEN2, ... so rewrite the list without gaps.
    foreach ($old in (Get-OpenEntries $Key)) { $Key.DeleteValue($old.Name) }
    $index = 0
    foreach ($entry in $Entries) {
        $Key.SetValue((Get-OpenName $index), $entry.Value, $entry.Kind)
        $index++
    }
}

function Register-AddIn([string]$XlamPath) {
    $key = Open-UserKey "$ExcelSubKey\Options" -Create
    try {
        $wanted = '/R "' + $XlamPath + '"'
        $old = @(Get-OpenEntries $key)
        $new = @()
        $placed = $false
        foreach ($entry in $old) {
            if (Test-OurAddIn $entry.Value) {
                if (-not $placed) {
                    $new += [pscustomobject]@{ Value = $wanted; Kind = [Microsoft.Win32.RegistryValueKind]::String }
                    $placed = $true
                }
            } else {
                $new += $entry
            }
        }
        if (-not $placed) { $new += [pscustomobject]@{ Value = $wanted; Kind = [Microsoft.Win32.RegistryValueKind]::String } }
        $unchanged = ($old.Count -eq $new.Count)
        for ($i = 0; $unchanged -and $i -lt $old.Count; $i++) {
            if (($old[$i].Name -ne (Get-OpenName $i)) -or ($old[$i].Value -ne $new[$i].Value)) { $unchanged = $false }
        }
        if ($unchanged) {
            Write-Info 'Add-in is already registered to load with Excel.'
        } else {
            Set-OpenEntries $key $new
            Write-Info "Registered the add-in to load with Excel ($wanted)."
        }
    } finally { $key.Close() }
    Remove-AddInManagerEntries
}

function Unregister-AddIn {
    $key = Open-UserKey "$ExcelSubKey\Options"
    if ($key) {
        try {
            $old = @(Get-OpenEntries $key)
            $keep = @($old | Where-Object { -not (Test-OurAddIn $_.Value) })
            if ($keep.Count -ne $old.Count) {
                Set-OpenEntries $key $keep
                Write-Info 'Removed the add-in from the list Excel loads at startup.'
            }
        } finally { $key.Close() }
    }
    Remove-AddInManagerEntries
}

function Remove-AddInManagerEntries {
    # Excel moves add-ins that were unticked in File > Options > Add-ins here.
    $key = Open-UserKey "$ExcelSubKey\Add-in Manager"
    if (-not $key) { return }
    try {
        foreach ($name in $key.GetValueNames()) {
            if (Test-OurAddIn $name) { $key.DeleteValue($name) }
        }
    } finally { $key.Close() }
}

function Get-OurTrustedLocations($Key, [string]$Root) {
    $ours = @()
    foreach ($sub in $Key.GetSubKeyNames()) {
        $location = $Key.OpenSubKey($sub)
        try {
            $path = [string]$location.GetValue('Path', '')
            $description = [string]$location.GetValue('Description', '')
        } finally { $location.Close() }
        if (($description -eq $AppDisplayName) -or ($Root -and ($path.TrimEnd('\') -ieq $Root.TrimEnd('\')))) { $ours += $sub }
    }
    return $ours
}

function Register-TrustedLocation([string]$Root) {
    $key = Open-UserKey "$ExcelSubKey\Security\Trusted Locations" -Create
    try {
        $ours = @(Get-OurTrustedLocations $key $Root)
        if ($ours.Count -gt 0) {
            $name = $ours[0]
            foreach ($extra in ($ours | Select-Object -Skip 1)) { $key.DeleteSubKeyTree($extra) }
        } else {
            $i = 0
            while ($key.GetSubKeyNames() -contains "Location$i") { $i++ }
            $name = "Location$i"
        }
        $location = $key.CreateSubKey($name)
        try {
            $location.SetValue('Path', ($Root.TrimEnd('\') + '\'), [Microsoft.Win32.RegistryValueKind]::String)
            $location.SetValue('Description', $AppDisplayName, [Microsoft.Win32.RegistryValueKind]::String)
        } finally { $location.Close() }
        Write-Info "Trusted location $name -> $Root"
    } finally { $key.Close() }
}

function Unregister-TrustedLocation([string]$Root) {
    $key = Open-UserKey "$ExcelSubKey\Security\Trusted Locations"
    if (-not $key) { return }
    try {
        foreach ($name in (Get-OurTrustedLocations $key $Root)) {
            $key.DeleteSubKeyTree($name)
            Write-Info "Removed trusted location $name."
        }
    } finally { $key.Close() }
}

function Register-UninstallEntry([string]$Root, [string]$Version) {
    $key = Open-UserKey $UninstallSubKey -Create
    try {
        $key.SetValue('DisplayName', $AppDisplayName, [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('DisplayVersion', $Version, [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('Publisher', $AppName, [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('InstallLocation', $Root, [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('UninstallString', ('"' + (Join-Path $Root 'uninstall.bat') + '"'), [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('InstallDate', (Get-Date -Format 'yyyyMMdd'), [Microsoft.Win32.RegistryValueKind]::String)
        $key.SetValue('NoModify', 1, [Microsoft.Win32.RegistryValueKind]::DWord)
        $key.SetValue('NoRepair', 1, [Microsoft.Win32.RegistryValueKind]::DWord)
    } finally { $key.Close() }
}


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

function Test-PythonVariables {
    # The Excel UDF server inherits the user's environment; these two break it.
    foreach ($name in 'PYTHONHOME', 'PYTHONPATH') {
        foreach ($scope in 'User', 'Machine') {
            if ([Environment]::GetEnvironmentVariable($name, $scope)) {
                Write-Warn "$name is set for the $($scope.ToLower()) environment. It can stop the Excel functions from starting; see README (Troubleshooting)."
            }
        }
    }
}

function Test-Excel {
    $appPaths = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\excel.exe', 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\excel.exe'
    if (-not (@($appPaths | Where-Object { Test-Path -LiteralPath $_ }))) {
        Write-Warn 'Desktop Microsoft Excel was not found. Extraplus is installed anyway.'
    }
    $c2r = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Office\ClickToRun\Configuration' -ErrorAction SilentlyContinue
    if ($c2r -and ($c2r.PSObject.Properties.Name -contains 'Platform') -and ($c2r.Platform -eq 'x86')) {
        Write-Warn '32-bit Office detected. 32-bit Office has not been tested.'
    }
}

function Install-Extraplus {
    $pins = Import-PowerShellDataFile -LiteralPath (Join-Path $SourceDir 'pins.psd1')
    $manifest = Get-Content -LiteralPath (Join-Path $SourceDir 'manifest.json') -Raw | ConvertFrom-Json

    Write-Host "$AppDisplayName $($manifest.version) - install / update"
    Assert-ExcelClosed
    Test-Excel
    Test-PythonVariables

    $root = Resolve-InstallRoot $InstallRoot
    Write-Step "Install folder: $root"
    $isNew = -not (Test-Path -LiteralPath $root)
    try {
        New-Item -ItemType Directory -Force -Path $root | Out-Null
    } catch {
        throw "Could not create $root ($($_.Exception.Message))."
    }
    if ($isNew -and -not (Test-InUserProfile $root)) {
        Protect-Folder $root
        Write-Info 'Limited the folder permissions to this user.'
    }
    Stop-ExtraplusProcesses $root
    Set-UvEnvironment $root
    Push-Location -LiteralPath $root
    try {
        Write-Step "uv $($pins.UvVersion)"
        $uv = Install-Uv $root $pins
        $venv = Install-PythonEnvironment $root $uv $pins $manifest

        Write-Step 'Excel add-in'
        $udf = Join-Path $root 'udf'
        New-Item -ItemType Directory -Force -Path $udf | Out-Null
        $copies = @(
            @{ From = 'extrapolation.py'; To = (Join-Path $udf 'extrapolation.py') },
            @{ From = $AddInFile; To = (Join-Path $root $AddInFile) },
            @{ From = 'install.ps1'; To = (Join-Path $root 'install.ps1') },
            @{ From = 'uninstall.ps1'; To = (Join-Path $root 'uninstall.ps1') },
            @{ From = 'uninstall.bat'; To = (Join-Path $root 'uninstall.bat') }
        )
        foreach ($copy in $copies) {
            Copy-Item -LiteralPath (Join-Path $SourceDir $copy.From) -Destination $copy.To -Force
            Unblock-File -LiteralPath $copy.To   # remove the "downloaded from the internet" mark
        }
        Write-Info "Copied the add-in and the UDF module to $root"

        Write-Step 'Checking the installation'
        $venvPython = Join-Path $venv 'Scripts\python.exe'
        $dll = Join-Path $venv "xlwings64-$($manifest.xlwings).dll"
        if (-not (Test-Path -LiteralPath $dll)) { throw "$dll is missing; the add-in needs xlwings $($manifest.xlwings)." }
        $env:PYTHONPATH = $udf
        try {
            $check = Get-NativeOutput $venvPython @('-c', "import importlib.metadata as m, extrapolation; print(extrapolation.EXTRAPLUS_VERSION(), m.version('xlwings'))")
        } finally {
            Remove-Item -LiteralPath 'Env:PYTHONPATH' -ErrorAction SilentlyContinue
        }
        $expected = "$($manifest.version) $($manifest.xlwings)"
        if ($check -ne $expected) { throw "Installed versions '$check' do not match '$expected'." }
        Write-Info "extraplus $($manifest.version) and xlwings $($manifest.xlwings) import correctly."

        Write-Step 'Registering with Excel'
        [Environment]::SetEnvironmentVariable($EnvVarName, $root, 'User')   # also notifies running programs
        $env:EXTRAPLUS_HOME = $root
        Write-Info "$EnvVarName = $root"
        Register-AddIn (Join-Path $root $AddInFile)
        Register-TrustedLocation $root
        Register-UninstallEntry $root $manifest.version

        $info = [ordered]@{
            version   = $manifest.version
            xlwings   = $manifest.xlwings
            python    = $pins.PythonVersion
            uv        = $pins.UvVersion
            installed = (Get-Date).ToString('s')
        }
        $info | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root $MarkerFile) -Encoding ASCII
    } finally {
        Pop-Location
    }

    Write-Host ''
    Write-Host "SUCCESS: $AppDisplayName $($manifest.version) is installed." -ForegroundColor Green
    Write-Host '    Open Excel (any workbook, .xlsx is fine) and type for example'
    Write-Host '        =EXTRAP1(A2:A20, B2:B20)'
    Write-Host '    The first calculation in each Excel session takes a few seconds.'
    Write-Host "    You can delete the folder you unzipped. To uninstall, use Settings > Apps > $AppDisplayName."
}


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

function Remove-InstallFolder([string]$Root) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $MarkerFile))) {
        throw "Refusing to delete $Root because it does not contain $MarkerFile."
    }
    if ($Root.TrimEnd('\').Length -lt 8) { throw "Refusing to delete $Root." }
    # rmdir removes the junctions uv creates without following them.
    & cmd.exe /d /c rmdir /s /q $Root
    if (Test-Path -LiteralPath $Root) {
        throw "Could not delete every file in $Root. Close all programs, sign out and back in, then run the uninstaller again."
    }
    Write-Info "Deleted $Root"
}

function Uninstall-Extraplus {
    Write-Host "$AppDisplayName - uninstall"
    Assert-ExcelClosed
    # Windows cannot delete a folder that is some process's working directory.
    Set-Location -LiteralPath $env:TEMP
    [Environment]::CurrentDirectory = $env:TEMP

    $roots = @()
    if ($InstallRoot) { $roots = @([IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')) } else { $roots = @(Find-InstalledRoots) }

    Write-Step 'Excel settings'
    Unregister-AddIn
    Unregister-TrustedLocation $null
    foreach ($root in $roots) { Unregister-TrustedLocation $root }
    if ([Environment]::GetEnvironmentVariable($EnvVarName, 'User')) {
        [Environment]::SetEnvironmentVariable($EnvVarName, $null, 'User')
        Write-Info "Removed $EnvVarName."
    }
    $entry = Open-UserKey $UninstallSubKey
    if ($entry) {
        $entry.Close()
        [Microsoft.Win32.Registry]::CurrentUser.DeleteSubKeyTree($UninstallSubKey)
        Write-Info 'Removed the Apps & features entry.'
    }

    Write-Step 'Files'
    if ($roots.Count -eq 0) { Write-Info 'No install folder found.' }
    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root) {
            Stop-ExtraplusProcesses $root
            Remove-InstallFolder $root
        }
    }

    Write-Host ''
    Write-Host "SUCCESS: $AppDisplayName has been removed." -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# Entry point (skipped when this file is dot-sourced by build\build_addin.ps1)
# ---------------------------------------------------------------------------

if ($MyInvocation.InvocationName -ne '.') {
    $exitCode = 0
    $log = Join-Path $env:TEMP 'extraplus-setup.log'
    try { Start-Transcript -LiteralPath $log -Force | Out-Null } catch { }
    try {
        if ($Uninstall) { Uninstall-Extraplus } else { Install-Extraplus }
    } catch {
        $exitCode = $FailureExitCode
        Write-Host ''
        Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "    Details are in $log"
    } finally {
        try { Stop-Transcript | Out-Null } catch { }
    }
    if (-not $NoPause) {
        Write-Host ''
        Read-Host 'Press Enter to close this window' | Out-Null
    }
    exit $exitCode
}
