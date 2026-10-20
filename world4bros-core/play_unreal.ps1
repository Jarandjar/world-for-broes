$repoRoot = $PSScriptRoot
if (-not (Test-Path (Join-Path $repoRoot "package.json"))) {
    $repoRoot = Join-Path $PSScriptRoot "world4bros-core"
}

Write-Host "[World4Bros] Launching Unreal Editor via unified CLI..." -ForegroundColor Cyan
Push-Location $repoRoot
try {
    node tools/run-unreal.mjs open
} finally {
    Pop-Location
}
