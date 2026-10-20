@echo off
set REPO=%~dp0
if not exist "%REPO%package.json" set REPO=%~dp0world4bros-core\

echo [World4Bros] Launching Unreal Editor via unified CLI...
pushd "%REPO%"
node tools\run-unreal.mjs open
popd
