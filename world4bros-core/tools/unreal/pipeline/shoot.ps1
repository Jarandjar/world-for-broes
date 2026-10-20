<#
  W4B screenshot runner: opens the editor off-screen, saves a PNG of a level from its W4B preview camera, closes.
  Success = the W4B_SHOT OK marker plus the file, not the exit code: the editor sometimes crashes while closing
  after the PNG is saved, and a start right after another Unreal process can fail, so it retries once.
  Tested on UE 5.7.3 and re-tested on 5.8.3 (docs/UE_PIPELINE.md section 6). First run compiles shaders (minutes); later runs ~20-30 s.

  powershell -File tools/unreal/pipeline/shoot.ps1 -Project C:/W4B/W4BArena/World4Bros.uproject -Map /Game/W4B/Maps/L_W4B_DuelPit -Out C:/W4B/shots/duel_pit.png
#>
param(
  [Parameter(Mandatory = $true)][string]$Project,
  [Parameter(Mandatory = $true)][string]$Map,
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Res = "1600x900",
  [string]$Engine = $(if ($env:W4B_UE_ROOT) { $env:W4B_UE_ROOT } else { 'C:\Program Files\Epic Games\UE_5.8' }),
  [int]$TimeoutSec = 1500
)
$exe = Join-Path $Engine 'Engine\Binaries\Win64\UnrealEditor.exe'
$script = (Join-Path $PSScriptRoot 'shoot_level.py').Replace('\', '/')
$projectPath = (Resolve-Path $Project).Path
$outPath = [IO.Path]::GetFullPath($Out).Replace('\', '/')
$logName = [IO.Path]::GetFileNameWithoutExtension($projectPath) + '.log'
$log = Join-Path (Split-Path $projectPath) "Saved\Logs\$logName"

for ($try = 1; $try -le 2; $try++) {
  if (Test-Path $outPath) { Remove-Item $outPath -Force }
  Get-Process UnrealEditor -ErrorAction SilentlyContinue | Wait-Process -Timeout 60 -ErrorAction SilentlyContinue
  $argLine = '"' + $projectPath + '" -ExecutePythonScript="' + $script + ' --map ' + $Map + ' --out ' + $outPath + ' --res ' + $Res + '" -RenderOffScreen -unattended -nosplash -nosound -log'
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $proc = Start-Process -FilePath $exe -ArgumentList $argLine -PassThru -WindowStyle Hidden
  if (-not $proc.WaitForExit($TimeoutSec * 1000)) { $proc.Kill(); Write-Output "W4B_SHOT timeout after $TimeoutSec s (try $try)"; continue }
  $sw.Stop()
  $ok = (Test-Path $outPath) -and (Select-String -Path $log -Pattern 'W4B_SHOT OK' -Quiet -ErrorAction SilentlyContinue)
  Write-Output ("W4B_SHOT try={0} ok={1} exit={2} seconds={3} file={4}" -f $try, $ok, $proc.ExitCode, [int]$sw.Elapsed.TotalSeconds, $outPath)
  if ($ok) { exit 0 }
  Start-Sleep -Seconds 5
}
exit 1
