<#
  W4B Unreal runner: runs one Python script inside Unreal without a window and returns Unreal's exit code.
  Prints only the lines that matter (W4B_* markers and errors). Full log: <Project>/Saved/Logs/<Project>.log
  Tested on UE 5.7.3 (docs/UE_PIPELINE.md section 6).

  Example:
    powershell -File tools/unreal/pipeline/ue.ps1 -Project C:/W4B/W4BArena/W4BArena.uproject -Script tools/unreal/pipeline/build_arena.py -ScriptArgs "--arena all"
    powershell -File tools/unreal/pipeline/ue.ps1 -Project ... -Script tools/unreal/pipeline/verify_arena.py
#>
param(
  [Parameter(Mandatory = $true)][string]$Project,
  [Parameter(Mandatory = $true)][string]$Script,
  [string]$ScriptArgs = "",
  [string]$Engine = $(if ($env:W4B_UE_ROOT) { $env:W4B_UE_ROOT } else { 'C:\Program Files\Epic Games\UE_5.7' }),
  [string]$Filter = 'LogPython: (W4B_|Error)|LogPython: Warning: W4B|Error: W4B|Assertion failed|Unhandled Exception|Fatal error|Python script executed with errors'
)
$exe = Join-Path $Engine 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
if (-not (Test-Path $exe)) { Write-Error "UnrealEditor-Cmd.exe not found under $Engine"; exit 2 }
$scriptPath = (Resolve-Path $Script).Path.Replace('\', '/')
$projectPath = (Resolve-Path $Project).Path
$inner = ($scriptPath + ' ' + $ScriptArgs).Trim()
$argLine = '"' + $projectPath + '" -run=pythonscript -script="' + $inner + '" -unattended -nullrhi -nosplash -nosound -stdout -FullStdOutLogOutput'

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.Arguments = $argLine
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = [System.Diagnostics.Process]::Start($psi)
$out = $proc.StandardOutput.ReadToEndAsync()
$err = $proc.StandardError.ReadToEndAsync()
$proc.WaitForExit()
$sw.Stop()
(($out.Result + "`n" + $err.Result) -split "`r?`n") | Select-String $Filter | ForEach-Object { $_.Line }
Write-Output ("W4B_RUN exit={0} seconds={1} script={2}" -f $proc.ExitCode, [int]$sw.Elapsed.TotalSeconds, (Split-Path $Script -Leaf))
exit $proc.ExitCode
