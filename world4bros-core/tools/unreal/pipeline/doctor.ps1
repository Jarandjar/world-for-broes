<#
  W4B Unreal doctor: checks this PC and the Unreal project against docs/UE_PIPELINE.md section 2-section 3.
  Prints PASS / WARN / FAIL with a fix for each problem; exit code 1 if anything FAILs.
  Run: npm run ue:doctor
#>
$ErrorActionPreference = 'SilentlyContinue'
$fails = 0
function Out-Check([string]$level, [string]$what, [string]$fix = '') {
  if ($level -eq 'FAIL') { $script:fails++ }
  $line = "{0,-5} {1}" -f $level, $what
  if ($fix) { $line += "`n      fix: $fix" }
  Write-Output $line
}

# Unreal Engine
$ueRoot = if ($env:W4B_UE_ROOT) { $env:W4B_UE_ROOT } else { 'C:\Program Files\Epic Games\UE_5.8' }
$ver = Get-Content (Join-Path $ueRoot 'Engine\Build\Build.version') -Raw | ConvertFrom-Json
if ($ver) { Out-Check 'PASS' ("Unreal Engine {0}.{1}.{2} at {3}" -f $ver.MajorVersion, $ver.MinorVersion, $ver.PatchVersion, $ueRoot) }
else { Out-Check 'FAIL' "Unreal Engine not found at $ueRoot" 'install UE 5.8 from the Epic Games Launcher, or set W4B_UE_ROOT' }

# Visual Studio C++ (allowed MSVC versions are read from the installed engine below)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) { & $vswhere -all -prerelease -format json | ConvertFrom-Json } else { @() }
$withGame = if (Test-Path $vswhere) { & $vswhere -all -prerelease -requires Microsoft.VisualStudio.Workload.NativeGame -property installationPath } else { @() }
if (-not $vs) { Out-Check 'FAIL' 'no Visual Studio found' 'install Visual Studio 2022 or 2026 with "Game development with C++" (docs/UE_PIPELINE.md section 2)' }
else {
  foreach ($i in $vs) { Out-Check 'INFO' ("{0} {1}" -f $i.displayName, $i.installationVersion) }
  if (-not $withGame) { Out-Check 'FAIL' 'no Visual Studio has the "Game development with C++" workload' 'Visual Studio Installer > Modify > tick "Game development with C++" and "Desktop development with C++" (exact components in docs/UE_PIPELINE.md section 2)' }
  else { Out-Check 'PASS' "Game development with C++ installed: $($withGame -join ', ')" }
}
# Preferred and banned MSVC versions come from the installed engine itself (Engine/Config/Windows/Windows_SDK.json).
$sdkJson = Get-Content (Join-Path $ueRoot 'Engine\Config\Windows\Windows_SDK.json') -Raw | ConvertFrom-Json
function In-Ranges([string]$v, $ranges) {
  foreach ($r in $ranges) { $lo, $hi = $r -split '-'; if ([version]$v -ge [version]$lo -and [version]$v -le [version]$hi) { return $true } }
  return $false
}
$msvc = foreach ($i in $vs) { Get-ChildItem (Join-Path $i.installationPath 'VC\Tools\MSVC') -Directory | Select-Object -ExpandProperty Name }
$banned = $msvc | Where-Object { In-Ranges $_ $sdkJson.BannedVisualCppVersions }
$good = $msvc | Where-Object { (In-Ranges $_ $sdkJson.PreferredVisualCppVersions) -and -not (In-Ranges $_ $sdkJson.BannedVisualCppVersions) }
$prefText = ($sdkJson.PreferredVisualCppVersions -join ' or ')
if ($good) { Out-Check 'PASS' "MSVC toolset preferred by this Unreal: $($good -join ', ')" }
elseif ($msvc) { Out-Check 'FAIL' "MSVC toolsets found ($($msvc -join ', ')) but none is preferred and allowed (this Unreal wants $prefText)" 'Visual Studio Installer > Modify > Individual components: tick the newest "MSVC ... build tools" of a preferred family (14.50 for UE 5.8; 14.44 must be 14.44.35211 or newer), or update Visual Studio so the existing one updates' }
else { Out-Check 'FAIL' 'no MSVC C++ compiler installed' "add an MSVC build tools component in the range $prefText" }
if ($banned) { Out-Check 'WARN' "this Unreal bans these installed MSVC versions and will skip them: $($banned -join ', ')" }
$sdks = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\Include' -Directory | Select-Object -ExpandProperty Name
if ($sdks -contains '10.0.22621.0' -or ($sdks | Where-Object { [version]$_ -ge [version]'10.0.22621.0' })) { Out-Check 'PASS' "Windows SDK: $($sdks -join ', ')" }
else { Out-Check 'FAIL' 'Windows 11 SDK 10.0.22621 (or newer) missing' 'add the "Windows 11 SDK (10.0.22621.0)" component' }

# Git + LFS
$git = (Get-Command git).Source
if (-not $git -and (Test-Path 'C:\Program Files\Git\cmd\git.exe')) { Out-Check 'WARN' 'git is installed but not on PATH for this shell' 'add C:\Program Files\Git\cmd to PATH (System > Environment Variables), then open a new terminal' }
elseif ($git) { Out-Check 'PASS' "git: $(& git --version)"; $lfs = & git lfs version; if ($lfs) { Out-Check 'PASS' "git lfs: $lfs" } else { Out-Check 'FAIL' 'git lfs missing' 'git lfs install' } }
else { Out-Check 'FAIL' 'git not installed' 'install Git for Windows (includes Git LFS)' }

# Python (pipeline helpers)
$py = (Get-Command python).Source
if ($py) { Out-Check 'PASS' "python: $(& python --version 2>&1)" } else { Out-Check 'WARN' 'python not on PATH' "use the engine's copy: $ueRoot\Engine\Binaries\ThirdParty\Python3\Win64\python.exe" }

# Long paths
$lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled).LongPathsEnabled
if ($lp -eq 1) { Out-Check 'PASS' 'Windows long paths enabled' } else { Out-Check 'WARN' 'Windows long paths are off' 'keep the project path short (C:\W4B\...)' }

# Project
$proj = $env:W4B_UE_PROJECT
if (-not $proj) { Out-Check 'FAIL' 'W4B_UE_PROJECT is not set' 'setx W4B_UE_PROJECT C:\W4B\W4BArena\World4Bros.uproject  (after creating the project, step U1)' }
elseif (-not (Test-Path $proj)) { Out-Check 'FAIL' "W4B_UE_PROJECT points at a missing file: $proj" 'create the project (step U1) or fix the path' }
else {
  $dir = Split-Path (Resolve-Path $proj).Path
  if ($dir -match 'OneDrive|Dropbox|Google Drive') { Out-Check 'FAIL' "project is inside a synced folder: $dir" 'move it to C:\W4B\ (OneDrive locks and re-uploads thousands of generated files)' } else { Out-Check 'PASS' "project folder: $dir" }
  if ($dir.Length -gt 60) { Out-Check 'WARN' "project path is $($dir.Length) characters" 'keep it under ~60 (Unreal adds long Intermediate paths)' }
  $ini = Get-Content (Join-Path $dir 'Config\DefaultEngine.ini') -Raw
  foreach ($k in 'W4BCover', 'W4BBulletPass', 'RuntimeGeneration=Dynamic', 'Interchange.FeatureFlags.Import.SyncToBrowser=0', 'bRemoteExecution=True') {
    if ($ini -match [regex]::Escape($k)) { Out-Check 'PASS' "DefaultEngine.ini has $k" } else { Out-Check 'FAIL' "DefaultEngine.ini lacks $k" 'merge tools/unreal/pipeline/DefaultEngine.W4B.ini into Config/DefaultEngine.ini' }
  }
}

# Disk
$drive = if ($proj -and (Test-Path $proj)) { (Resolve-Path $proj).Path.Substring(0, 1) } else { 'C' }
$free = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
if ($free -lt 25) { Out-Check 'FAIL' "only $free GB free on ${drive}:" 'free space: empty the recycle bin, remove old downloads; Unreal needs ~25-45 GB for the demo (docs/UE_PIPELINE.md section 2)' }
elseif ($free -lt 45) { Out-Check 'WARN' "$free GB free on ${drive}: (tight; the demo needs ~25-45 GB)" }
else { Out-Check 'PASS' "$free GB free on ${drive}:" }

Write-Output ""
if ($fails) { Write-Output "W4B_DOCTOR $fails problem(s) to fix"; exit 1 } else { Write-Output 'W4B_DOCTOR all good'; exit 0 }
